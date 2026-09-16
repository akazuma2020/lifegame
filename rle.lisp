;;;; RLEファイルをGPU用の盤面データへ変換する
;;;;
;;;; RLE（Run Length Encoding）は、同じものが何個続くかを数字で表す圧縮方法です。
;;;; LifeのRLEでは、たとえば「3o2b$」は「生セル3個、死セル2個、次の行」を表します。
;;;;   o = alive（生セル）  b = blank/dead（死セル）
;;;;   $ = 次の行          ! = データ終端
;;;;
;;;; GPUへ渡すときは、横32セルを1個のu32（32ビット整数）へ詰めます。
;;;; 1ビットが1セルに対応し、1なら生、0なら死です。

;;; packageは、同じ名前の変数や関数が別プログラムと衝突しないための「名前空間」。
(defpackage :lifegame
  (:use :cl)
  (:export :main :run-main-thread :parse-rle-into-grid))
(in-package :lifegame)

;;; -------------------- 盤面とタイルの寸法 --------------------
(defconstant +grid-width+ 20000)
(defconstant +grid-height+ 20000)
;;; ceilingは切り上げ。横20,000セルを32セルずつに分けるので625 words/rowになる。
(defconstant +words-per-row+ (ceiling +grid-width+ 32))
(defconstant +word-count+ (* +words-per-row+ +grid-height+))
(defconstant +tile-core-words+ 4)
(defconstant +tile-core-rows+ 128)
(defconstant +tile-columns+ (ceiling +words-per-row+ +tile-core-words+))
(defconstant +tile-rows+ (ceiling +grid-height+ +tile-core-rows+))
(defconstant +tile-count+ (* +tile-columns+ +tile-rows+))

(defun integer-after (marker text)
  "TEXT内でMARKERの直後にある整数を読む。RLEヘッダーの幅と高さに使う。"
  (let ((position (search marker text :test #'char-equal)))
    (and position
         (parse-integer text :start (+ position (length marker)) :junk-allowed t))))

(defun read-rle (pathname)
  "Return WIDTH, HEIGHT and a compact string containing the RLE body."
  ;; コメント行（#で始まる行）と空行を飛ばし、ヘッダーと本体を分ける。
  (let ((width nil) (height nil))
    (values
     (with-output-to-string (body)
       (with-open-file (input pathname :direction :input)
         (loop for line = (read-line input nil nil)
               while line do
                 (cond
                   ;; condの各節は「もし～なら」を上から順に試す。ここは何もしない節。
                   ((or (zerop (length line)) (char= (char line 0) #\#)))
                   ((and (null width) (search "x" line :test #'char-equal)
                         (search "=" line))
                    (setf width (integer-after "x =" line)
                          height (integer-after "y =" line))
                    (unless (and width height)
                      (error "Invalid RLE header: ~A" line)))
                   ;; ヘッダー以外の行は、改行を除いて1本のRLE文字列へつなぐ。
                   (t (write-string line body))))))
     width height)))

;;; 小さな関数なのでinline化を処理系へお願いする。セル数が多いと呼び出し回数も多いため。
(declaim (inline set-live-cell))
(defun set-live-cell (words x y)
  ;; x/32で格納先u32を選び、x mod 32でその中のビット位置を選ぶ。
  ;; ash x -5 は整数の x/32、logand x 31 は x mod 32 と同じ意味になる。
  (let* ((index (+ (* y +words-per-row+) (ash x -5)))
         (mask (ash 1 (logand x 31))))
    (setf (aref words index) (logior (aref words index) mask))))

(declaim (inline mark-occupied-tile))
(defun mark-occupied-tile (occupied x y)
  ;; 生セルを見つけるたびに、それが属するタイルへ印を付ける。
  ;; この段階では周囲のタイルへはまだ広げない。
  (let ((tile-x (floor (ash x -5) +tile-core-words+))
        (tile-y (floor y +tile-core-rows+)))
    (setf (sbit occupied (+ (* tile-y +tile-columns+) tile-x)) 1)))

(defun make-active-tile-list (occupied)
  "Expand occupied tiles by one tile in every direction and return their indices."
  ;; Lifeは隣のセルへ1世代で影響を伝える。生セル入りタイルだけでなく、周囲8タイルも
  ;; 次の計算候補にしておけば、タイル境界を越えて誕生するセルを見落とさない。
  (let ((candidates (make-array +tile-count+ :element-type 'bit :initial-element 0)))
    (dotimes (tile-index +tile-count+)
      (when (= (sbit occupied tile-index) 1)
        (multiple-value-bind (tile-y tile-x)
            (floor tile-index +tile-columns+)
          (loop for dy from -1 to 1 do
            (loop for dx from -1 to 1
                  for x = (+ tile-x dx)
                  for y = (+ tile-y dy)
                  when (and (<= 0 x) (< x +tile-columns+)
                            (<= 0 y) (< y +tile-rows+))
                    do (setf (sbit candidates
                                    (+ (* y +tile-columns+) x))
                             1))))))
    ;; bit配列は印を付けるのに便利だが、GPUには印のある番号だけを昇順で渡す。
    (let* ((count (count 1 candidates))
           (tiles (make-array count :element-type '(unsigned-byte 32)))
           (output-index 0))
      (dotimes (tile-index +tile-count+ tiles)
        (when (= (sbit candidates tile-index) 1)
          (setf (aref tiles output-index) tile-index)
          (incf output-index))))))

(defun parse-rle-into-grid (pathname &key
                                      (grid-width +grid-width+)
                                      (grid-height +grid-height+))
  "Center an RLE pattern in a packed 20000-square grid.
Returns WORDS, LIVE-COUNT, PATTERN-WIDTH, PATTERN-HEIGHT and ACTIVE-TILES."
  ;;; -------------------- RLE本体を左上から順に読む --------------------
  ;; GPU側の横幅は固定値を前提にしているので、別の横幅を誤って渡さないよう確認する。
  (unless (= grid-width +grid-width+)
    (error "This GPU layout requires a grid width of ~D" +grid-width+))
  (multiple-value-bind (body pattern-width pattern-height) (read-rle pathname)
    (unless (and pattern-width pattern-height)
      (error "No RLE header found in ~A" pathname))
    (when (or (> pattern-width grid-width) (> pattern-height grid-height))
      (error "RLE pattern ~Dx~D does not fit the ~Dx~D grid"
             pattern-width pattern-height grid-width grid-height))
    (let* ((words (make-array (* +words-per-row+ grid-height)
                              :element-type '(unsigned-byte 32)
                              :initial-element 0))
           ;; パターンの左右・上下へ同じくらい余白を置き、20,000²の中央へ配置する。
           (offset-x (floor (- grid-width pattern-width) 2))
           (offset-y (floor (- grid-height pattern-height) 2))
           (occupied (make-array +tile-count+ :element-type 'bit :initial-element 0))
           ;; x,yはRLEパターン内の読取位置。runは直前までに読んだ桁の数値。
           (x 0) (y 0) (run 0) (live-count 0))
      ;; 数字が省略された場合は1個とみなし、使ったrunは0へ戻す。
      (labels ((take-run () (prog1 (if (zerop run) 1 run) (setf run 0))))
        (loop for character across body do
          (cond
            ((digit-char-p character)
             ;; 「123」を1,2,3と読んで ((1*10+2)*10+3) にする。
             (setf run (+ (* run 10) (digit-char-p character))))
            ((or (char-equal character #\b) (char-equal character #\o))
             (let ((count (take-run)))
               (when (char-equal character #\o)
                 ;; 生セルだけを書き込む。死セルは配列の初期値0のままでよい。
                 (when (or (> (+ x count) pattern-width) (>= y pattern-height))
                   (error "RLE body exceeds its declared dimensions near (~D,~D)" x y))
                 (dotimes (i count)
                   (let ((grid-x (+ offset-x x i))
                         (grid-y (+ offset-y y)))
                     (set-live-cell words grid-x grid-y)
                     (mark-occupied-tile occupied grid-x grid-y)))
                 (incf live-count count))
               (incf x count)))
            ((char= character #\$)
             ;; 指定された行数だけ下へ進み、横位置を行頭へ戻す。
             (incf y (take-run))
             (setf x 0)
             (when (> y pattern-height)
               (error "RLE body exceeds its declared height")))
            ;; !を読めば、後ろに文字があってもRLEデータは終了。
            ((char= character #\!) (return))
            ((find character '(#\Space #\Tab #\Return #\Newline)))
            (t (error "Unexpected RLE character ~S" character)))))
      ;; Common Lispは複数の戻り値を返せる。main側がmultiple-value-bindで受け取る。
      (values words live-count pattern-width pattern-height
              (make-active-tile-list occupied)))))
