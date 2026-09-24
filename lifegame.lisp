(declaim (optimize (speed 3) (safety 0) (debug 0)))
(ql:quickload :cffi)

;; 実行時ライブラリはOSの標準ローダー検索経路から解決する。
;; プロジェクト内のbridge/や環境固有のSDL2ディレクトリは検索対象に加えない。
(defparameter cl-user::*lifegame-bootstrap-source-directory*
  (uiop:pathname-directory-pathname
   (or *load-truename* *compile-file-truename*)))

(ql:quickload '(:sdl2 :bordeaux-threads))

;;;; WGPU Lifeの制御側
;;;;
;;;; rle.lispがRLEをGPU用のビット配列へ変換し、このファイルがSDLの入力、
;;;; 時計との同期、CFFI経由のGPU呼び出しを担当する。

(load (merge-pathnames
       "rle.lisp"
       (uiop:pathname-directory-pathname
        (or *load-truename* *compile-file-truename*))))
(in-package :lifegame)

(defparameter *source-directory*
  cl-user::*lifegame-bootstrap-source-directory*)

;;; -------------------- 表示・更新・時計の定数 --------------------
(defconstant +max-steps-per-frame+ 256)
(defconstant +sync-steps-per-batch+ 1024)
(defconstant +min-zoom-level+ -14)
(defconstant +max-zoom-level+ 19)
(defconstant +initial-zoom-level+ 16)

;; このLife時計は11,520世代で表示が1分進む。
(defconstant +clock-generations-per-minute+ 11520)
(defconstant +clock-generations-per-second+
  (/ +clock-generations-per-minute+ 60))
(defconstant +clock-snapshot-minutes+ 10)
(defconstant +clock-snapshot-interval+
  (* +clock-generations-per-minute+ +clock-snapshot-minutes+))
(defconstant +clock-snapshot-base-generation+ 24883200)
(defconstant +clock-generations-per-day+
  (* 24 60 +clock-generations-per-minute+))
(defconstant +clock-display-lag-generations+
  +clock-generations-per-minute+)
(defconstant +clock-correction-seconds+ 300)

(defparameter *clock-url*
  "https://gist.githubusercontent.com/anonymous/9d7468755dd76a35d93beeb5c0a5bdcf/raw/3295717faf24e8911048bcb69d4b6c8505d24330/gistfile1.txt")
(defparameter *clock-path* (merge-pathnames "clock-ampm.rle" *source-directory*))
(defparameter *clock-snapshot-directory*
  (merge-pathnames "clock-snapshot-ampm/" *source-directory*))

(defparameter *windows-platform-p* (uiop:os-windows-p))
(defparameter *video-driver*
  (or (uiop:getenv "LIFEGAME_SDL_VIDEODRIVER")
      ;; WSLではWaylandより検証済みのX11を既定にする。Windowsネイティブでは
      ;; SDL自身にWin32 driverを選ばせるため、環境変数を設定しない。
      (unless *windows-platform-p* "x11")))
(when *video-driver*
  (setf (uiop:getenv "SDL_VIDEODRIVER") *video-driver*))
(setf (uiop:getenv "RUST_BACKTRACE") "full")

;;; -------------------- C bridgeとの接続 --------------------
(if *windows-platform-p*
    (progn
      ;; DLLはWindowsの標準DLL検索経路から名前だけで解決する。
      (cffi:load-foreign-library "wgpu_native.dll")
      (cffi:load-foreign-library "bridge/bridge.dll"))
    (progn
      ;; WSL固有のD3D12 libraryは存在するときだけ先にloadする。これにより
      ;; 同じソースを通常のLinuxでもloadできる。
      (dolist (path '("/usr/lib/wsl/lib/libd3d12core.so"
                      "/usr/lib/wsl/lib/libd3d12.so"))
        (when (probe-file path)
          (cffi:load-foreign-library path)))
      ;; 共有ライブラリはOSの標準検索経路から名前だけで解決する。
      (cffi:load-foreign-library "libwgpu_native.so")
      (cffi:load-foreign-library "bridge/bridge.so")))

(cffi:defcfun ("WGPU_CreateInstance" %create-instance) :pointer
  (allow-noncompliant :int))
(cffi:defcfun ("WGPU_InitSurface" init-surface) :pointer
  (instance :pointer) (window :pointer))
(cffi:defcfun ("WGPU_InitLife" %init-life) :pointer
  (instance :pointer) (surface :pointer) (shader :string)
  (initial-words :pointer) (word-count :uint64)
  (initial-active-tiles :pointer) (initial-tile-count :uint32)
  (out-state :pointer))
(cffi:defcfun ("WGPU_UpdateSurface" update-surface) :int
  (device :pointer) (surface :pointer)
  (width :uint32) (height :uint32))
(cffi:defcfun ("WGPU_SetPresentMode" set-present-mode) :int
  (device :pointer) (surface :pointer) (state :pointer)
  (width :uint32) (height :uint32) (request-immediate :uint32))
(cffi:defcfun ("WGPU_ReleaseLife" release-life) :void (state :pointer))
(cffi:defcfun ("wgpuDeviceGetQueue" get-queue) :pointer (device :pointer))
(cffi:defcfun ("wgpuQueueRelease" release-queue) :void (queue :pointer))
(cffi:defcfun ("wgpuDeviceRelease" release-device) :void (device :pointer))
(cffi:defcfun ("wgpuSurfaceRelease" release-surface) :void (surface :pointer))
(cffi:defcfun ("wgpuSurfaceUnconfigure" unconfigure-surface) :void
  (surface :pointer))
(cffi:defcfun ("wgpuInstanceRelease" release-instance) :void
  (instance :pointer))
(cffi:defcfun ("SDL_SetWindowTitle" %set-window-title) :void
  (window :pointer) (title :string))
(cffi:defcfun ("SDL_ShowWindow" %show-window) :void (window :pointer))


(defun require-handle (handle description)
  (when (cffi:null-pointer-p handle)
    (error "~A failed" description))
  handle)

;;; -------------------- GPU実験で共通に使う短いbridge API --------------------
;; Lispは「何をするか」をcommand配列に書く。
;; bridge.cは、その配列をまとめてwgpu-nativeへ渡す。
;; こうすると、LispとCを何度も往復せずにすむ。

(cffi:defcstruct life-state
  ;; この並びはbridge.cのLifeStateと同じにする。
  (render-pipeline :pointer)
  (step-one-pipeline :pointer)
  (step-eight-pipeline :pointer)
  (clear-pipeline :pointer)
  (uniforms :pointer)
  (cells :pointer :count 2)
  (tile-flags :pointer :count 2)
  (active-tiles :pointer :count 2)
  (indirect-args :pointer :count 2)
  (render-groups :pointer :count 2)
  (step-one-groups :pointer :count 2)
  (step-eight-groups :pointer :count 2)
  (clear-groups :pointer :count 2)
  (current :uint32)
  (present-mode :uint32)
  (immediate-supported :boolean))

(cffi:defcstruct wgpu-bridge-compute-command
  ;; opcodeは命令の種類。残りのfieldは命令に必要な値を入れる箱。
  (opcode :uint32)
  (x :uint32)
  (y :uint32)
  (z :uint32)
  (offset :uint64)
  (size :uint64)
  (object :pointer)
  (argument :pointer))

(cffi:defcfun ("WGPU_RunComputeCommands" run-compute-commands) :int
  (device :pointer) (queue :pointer)
  (commands :pointer) (command-count :uint32) (wait :uint32))

(cffi:defcfun ("WGPU_RunComputeRenderFrame" run-compute-render-frame) :int
  (device :pointer) (queue :pointer) (surface :pointer)
  (commands :pointer) (command-count :uint32)
  (pipeline :pointer) (bind-group :pointer)
  (clear-r :double) (clear-g :double) (clear-b :double) (clear-a :double))

(cffi:defcfun ("wgpuQueueWriteBuffer" wgpu-queue-write-buffer) :void
  (queue :pointer) (buffer :pointer) (buffer-offset :uint64)
  (data :pointer) (size :size))

(defconstant +uniform-bytes+ 48)
(defconstant +grid-bytes+ (* +word-count+ 4))
(defconstant +tile-buffer-bytes+ (* +tile-count+ 4))
(defconstant +indirect-bytes+ 12)

;; bridge.cと同じ命令番号。
(defconstant +compute-clear-buffer+ 1)
(defconstant +compute-begin-pass+ 2)
(defconstant +compute-set-pipeline+ 3)
(defconstant +compute-set-bind-group+ 4)
(defconstant +compute-dispatch+ 5)
(defconstant +compute-dispatch-indirect+ 6)
(defconstant +compute-end-pass+ 7)

(defun state-slot (state slot &optional index)
  "LifeStateからGPU handleや数値を1個読む。"
  (if index
      (cffi:mem-aref
       (cffi:foreign-slot-pointer state '(:struct life-state) slot)
       :pointer index)
      (cffi:foreign-slot-value state '(:struct life-state) slot)))

(defun (setf state-slot) (value state slot &optional index)
  "LifeStateへGPU handleや数値を1個書く。"
  (if index
      (setf (cffi:mem-aref
             (cffi:foreign-slot-pointer state '(:struct life-state) slot)
             :pointer index)
            value)
      (setf (cffi:foreign-slot-value state '(:struct life-state) slot) value)))

(defun write-uniforms (queue state width height center-x center-y zoom sparse-mode)
  "画面の大きさやcamera位置など、shaderが毎frame読む値を送る。"
  (cffi:with-foreign-object (data :uint8 +uniform-bytes+)
    (dotimes (i +uniform-bytes+)
      (setf (cffi:mem-aref data :uint8 i) 0))
    (setf (cffi:mem-ref data :float 0) width
          (cffi:mem-ref data :float 4) height
          (cffi:mem-ref data :float 8) zoom
          (cffi:mem-ref data :float 16) center-x
          (cffi:mem-ref data :float 20) center-y
          (cffi:mem-ref data :uint32 32) +grid-width+
          (cffi:mem-ref data :uint32 36) +grid-height+
          (cffi:mem-ref data :uint32 40) +words-per-row+
          (cffi:mem-ref data :uint32 44) (if sparse-mode 1 0))
    (wgpu-queue-write-buffer
     queue (state-slot state 'uniforms) 0 data +uniform-bytes+)))

(defun command-at (commands index)
  "command配列のINDEX番目を指すpointerを返す。"
  (cffi:mem-aptr commands '(:struct wgpu-bridge-compute-command) index))

(defun set-command (commands index opcode
                    &key (x 0) (y 0) (z 0) (offset 0) (size 0)
                         (object (cffi:null-pointer)))
  "commandを1個作る。使わないfieldには0かNULLを入れる。"
  (let ((command (command-at commands index)))
    (setf (cffi:foreign-slot-value
           command '(:struct wgpu-bridge-compute-command) 'opcode) opcode
          (cffi:foreign-slot-value
           command '(:struct wgpu-bridge-compute-command) 'x) x
          (cffi:foreign-slot-value
           command '(:struct wgpu-bridge-compute-command) 'y) y
          (cffi:foreign-slot-value
           command '(:struct wgpu-bridge-compute-command) 'z) z
          (cffi:foreign-slot-value
           command '(:struct wgpu-bridge-compute-command) 'offset) offset
          (cffi:foreign-slot-value
           command '(:struct wgpu-bridge-compute-command) 'size) size
          (cffi:foreign-slot-value
           command '(:struct wgpu-bridge-compute-command) 'object) object
          (cffi:foreign-slot-value
           command '(:struct wgpu-bridge-compute-command) 'argument)
          (cffi:null-pointer))))

(defun life-batch-count (steps)
  "8世代ずつ進め、最後に余った世代を1世代ずつ進める回数。"
  (+ (floor steps 8) (mod steps 8)))

(defun life-command-count (steps sparse-mode)
  "必要なcommand数を先に数え、ちょうどよい大きさの配列を作れるようにする。"
  (* (life-batch-count steps) (if sparse-mode 12 7)))

(defun fill-life-commands (commands state steps sparse-mode)
  "Lifeの更新手順を汎用command配列へ書き、次の面番号を返す。"
  (let ((index 0)
        (current (state-slot state 'current)))
    (labels ((emit (opcode &rest arguments)
               (apply #'set-command commands index opcode arguments)
               (incf index)))
      (loop while (plusp steps)
            for batch = (if (>= steps 8) 8 1)
            for next = (logxor current 1)
            for pipeline =
              (state-slot state
                          (if (= batch 8)
                              'step-eight-pipeline
                              'step-one-pipeline))
            for groups =
              (if (= batch 8) 'step-eight-groups 'step-one-groups)
            do
               ;; 次に使う面へ残っている古い印を消す。
               (emit +compute-clear-buffer+
                     :object (state-slot state 'tile-flags next)
                     :size +tile-buffer-bytes+)

               ;; SPARSEでは、前回使ったtileだけを先に空にする。
               (when sparse-mode
                 (emit +compute-begin-pass+)
                 (emit +compute-set-pipeline+
                       :object (state-slot state 'clear-pipeline))
                 (emit +compute-set-bind-group+
                       :object (state-slot state 'clear-groups next))
                 (emit +compute-dispatch-indirect+
                       :object (state-slot state 'indirect-args next))
                 (emit +compute-end-pass+))

               ;; 次の候補数を0へ戻してからLifeを計算する。
               (emit +compute-clear-buffer+
                     :object (state-slot state 'indirect-args next)
                     :size 4)
               (emit +compute-begin-pass+)
               (emit +compute-set-pipeline+ :object pipeline)
               (emit +compute-set-bind-group+
                     :object (state-slot state groups current))
               (if sparse-mode
                   (emit +compute-dispatch-indirect+
                         :object (state-slot state 'indirect-args current))
                   (emit +compute-dispatch+ :x +tile-count+ :y 1 :z 1))
               (emit +compute-end-pass+)

               (setf current next)
               (decf steps batch))
      (values index current))))

(defun call-with-life-commands (state steps sparse-mode function)
  "Life commandを作り、FUNCTIONへ渡す。成功したときだけ面番号を進める。"
  (let ((count (life-command-count steps sparse-mode)))
    (if (zerop count)
        (funcall function (cffi:null-pointer) 0
                 (state-slot state 'current))
        (cffi:with-foreign-object
            (commands '(:struct wgpu-bridge-compute-command) count)
          (multiple-value-bind (actual-count next-current)
              (fill-life-commands commands state steps sparse-mode)
            (unless (= actual-count count)
              (error "Life command count mismatch: expected ~D, got ~D"
                     count actual-count))
            (let ((result (funcall function commands count next-current)))
              (when (zerop result)
                (setf (state-slot state 'current) next-current))
              result))))))

(defun advance-life (device queue state steps sparse-mode)
  "描画せずにSTEPS世代進める。起動時の早送りでも使う。"
  (write-uniforms queue state 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0
                  (not (zerop sparse-mode)))
  (call-with-life-commands
   state steps (not (zerop sparse-mode))
   (lambda (commands count next-current)
     (declare (ignore next-current))
     (run-compute-commands device queue commands count 1))))

(defun reset-life (queue state initial-words active-tiles)
  "最初の盤面と候補tileを、GPUの2面へ同じように入れ直す。"
  (let ((count (length active-tiles)))
    (cffi:with-foreign-object (indirect :uint32 3)
      (setf (cffi:mem-aref indirect :uint32 0) count
            (cffi:mem-aref indirect :uint32 1) 1
            (cffi:mem-aref indirect :uint32 2) 1)
      (cffi:with-pointer-to-vector-data (words-pointer initial-words)
        (dotimes (i 2)
          (wgpu-queue-write-buffer
           queue (state-slot state 'cells i) 0 words-pointer +grid-bytes+)
          (wgpu-queue-write-buffer
           queue (state-slot state 'indirect-args i) 0
           indirect +indirect-bytes+)))
      (when (plusp count)
        (cffi:with-pointer-to-vector-data (tiles-pointer active-tiles)
          (dotimes (i 2)
            (wgpu-queue-write-buffer
             queue (state-slot state 'active-tiles i) 0
             tiles-pointer (* count 4))))))
    (setf (state-slot state 'current) 0)))

(defun draw-life (device queue surface state width height center-x center-y zoom
                  steps sparse-mode)
  "Lifeを進め、その結果を1frame描いて画面へ出す。"
  (let ((sparse-p (not (zerop sparse-mode))))
    (write-uniforms queue state width height center-x center-y zoom sparse-p)
    (let ((result
            (call-with-life-commands
             state steps sparse-p
             (lambda (commands count next-current)
               (run-compute-render-frame
                device queue surface commands count
                (state-slot state 'render-pipeline)
                (state-slot state 'render-groups next-current)
                0.008d0 0.011d0 0.015d0 1.0d0)))))
      ;; Surfaceが古くなったときは、次のframeに備えて設定し直す。
      (when (= result 1)
        (set-present-mode device surface state
                          (truncate width) (truncate height)
                          (if (= (state-slot state 'present-mode) 3) 1 0)))
      result)))

;;; -------------------- 実行時設定 --------------------
(defparameter *window-title* "WGPU Life 20000 x 20000")
(defparameter *window-width* 1200)
(defparameter *window-height* 800)
(defparameter *window-x* 80)
(defparameter *window-y* 80)

(defparameter *speed-levels*
  #(1.0d0 2.0d0 3.0d0 4.0d0 5.0d0
    10.0d0 20.0d0 30.0d0 40.0d0 50.0d0 60.0d0 70.0d0 80.0d0 90.0d0 100.0d0
    110.0d0 120.0d0 130.0d0 140.0d0 150.0d0 160.0d0 170.0d0 180.0d0 190.0d0 192.0d0 200.0d0
    300.0d0 400.0d0 500.0d0 600.0d0 700.0d0 800.0d0 900.0d0 1000.0d0
    1100.0d0 1200.0d0 1300.0d0 1400.0d0 1500.0d0 1600.0d0 1700.0d0 1800.0d0 1900.0d0 2000.0d0
    2100.0d0 2200.0d0 2300.0d0 2400.0d0 2500.0d0 2600.0d0 2700.0d0 2800.0d0 2900.0d0 3000.0d0
    4000.0d0 5000.0d0 6000.0d0 7000.0d0 8000.0d0 9000.0d0 10000.0d0))

(defparameter *initial-speed* 50.0d0)
(defparameter *sparse-mode* t)
(defparameter *present-mode* :fifo)
(defparameter *sync-to-local-time* t)
(defvar *main-thread* nil)
(defparameter *auto-start*
  (not (string= (or (uiop:getenv "LIFEGAME_NO_AUTOSTART") "") "1")))


;;; -------------------- RLEと時計スナップショット --------------------
(defun ensure-clock-rle ()
  "Download the linked clock RLE once. Return its pathname."
  (unless (probe-file *clock-path*)
    (let ((temporary (merge-pathnames "clock.rle.download"
                                      *source-directory*)))
      (format t "~&Downloading the digital clock RLE...~%  ~A~%" *clock-url*)
      (handler-case
          (progn
            (uiop:run-program
             (list "curl" "-L" "--fail" "--silent" "--show-error"
                   "--output" (namestring temporary) *clock-url*)
             :output *standard-output* :error-output *error-output*)
            (rename-file temporary *clock-path*))
        (error (condition)
          (when (probe-file temporary)
            (delete-file temporary))
          (error "Clock RLE download failed: ~A~%Run ./fetch-clock.sh or save the linked RLE as ~A"
                 condition (namestring *clock-path*))))))
  *clock-path*)

(defun snapshot-generation (pathname)
  "Read the generation recorded in a Hashlife snapshot."
  (with-open-file (input pathname :direction :input)
    (loop for line = (read-line input nil nil)
          while line
          for marker = (search "#C Generation=" line)
          when marker do
            (return
              (parse-integer line
                             :start (+ marker (length "#C Generation="))
                             :junk-allowed t))
          finally
            (error "Snapshot has no Generation comment: ~A" pathname))))

(defun clock-snapshot-info (universal-time)
  "Return pathname, generation and wall-clock origin for the preceding 10-minute snapshot."
  ;; timezoneを指定しないdecode-universal-timeはOS設定のローカル時刻を返す。
  (multiple-value-bind (second minute hour)
      (decode-universal-time universal-time)
    (let* ((snapshot-minute
             (* (floor minute +clock-snapshot-minutes+)
                +clock-snapshot-minutes+))
           (snapshot-index
             (+ (* hour (/ 60 +clock-snapshot-minutes+))
                (/ snapshot-minute +clock-snapshot-minutes+)))
           (expected-generation
             (+ +clock-snapshot-base-generation+
                (* snapshot-index +clock-snapshot-interval+)))
           (snapshot-time
             (- universal-time
                second
                (* 60 (mod minute +clock-snapshot-minutes+))))
           (pathname
             (merge-pathnames
              (format nil "clock-~2,'0D-~2,'0D.rle" hour snapshot-minute)
              *clock-snapshot-directory*)))
      (unless (probe-file pathname)
        (error "Clock snapshot is missing: ~A~%Unzip clock-snapshot.zip or run hashlife/generate-snapshots.sh."
               pathname))
      (let ((actual-generation (snapshot-generation pathname)))
        (unless (= actual-generation expected-generation)
          (error "Wrong clock snapshot generation in ~A: expected ~:D, got ~:D"
                 pathname expected-generation actual-generation)))
      (list pathname expected-generation snapshot-time))))

(defun clock-target-generation (wall-time origin-time origin-generation)
  "Return the startup synchronization target without wrapping at midnight."
  (+ origin-generation
     (* (- wall-time origin-time) +clock-generations-per-second+)
     +clock-display-lag-generations+))

(defun local-generation-at (universal-time)
  "Return the generation phase corresponding to the local wall clock."
  (multiple-value-bind (second minute hour)
      (decode-universal-time universal-time)
    (+ +clock-snapshot-base-generation+
       (* (+ second (* 60 (+ minute (* 60 hour))))
          +clock-generations-per-second+)
       +clock-display-lag-generations+)))

(defun clock-generation-difference (target current)
  "Return the shortest signed TARGET-CURRENT difference on a 24-hour clock."
  (let ((difference (mod (- target current) +clock-generations-per-day+))
        (half-day (/ +clock-generations-per-day+ 2)))
    (if (> difference half-day)
        (- difference +clock-generations-per-day+)
        difference)))

;;; -------------------- UIの小さな補助関数 --------------------
(defun zoom-for-level (level)
  (min 64.0f0 (max 0.05f0 (expt 1.25f0 level))))

(defun full-view-zoom-level (width height)
  (let ((required-zoom
          (* 1.04f0 (max (/ (float +grid-width+ 1.0f0) width)
                            (/ (float +grid-height+ 1.0f0) height)))))
    (loop for level from +min-zoom-level+ to +max-zoom-level+
          when (>= (zoom-for-level level) required-zoom)
            return level
          finally (return +max-zoom-level+))))

(defun reset-gpu-grid (queue state words active-tiles)
  (reset-life queue state words active-tiles))

(defun scancode-is (keysym code)
  (sdl2:scancode= (sdl2:scancode-value keysym) code))

(defun next-speed-level (speed)
  (or (find-if (lambda (level) (> level speed)) *speed-levels*)
      (aref *speed-levels* (1- (length *speed-levels*)))))

(defun previous-speed-level (speed)
  (or (find-if (lambda (level) (< level speed))
               *speed-levels* :from-end t)
      (aref *speed-levels* 0)))

(defun configure-surface (device surface state width height present-mode)
  "Configure the surface and return the mode actually selected by wgpu-native."
  (unless (member present-mode '(:fifo :immediate))
    (error "Unknown present mode: ~S" present-mode))
  (let ((result (set-present-mode device surface state width height
                                  (if (eq present-mode :immediate) 1 0))))
    (when (minusp result)
      (error "Configure surface failed"))
    (if (plusp result)
        (progn
          (format t "~&Immediate present mode is unavailable; using FIFO.~%")
          (finish-output)
          :fifo)
        present-mode)))

;;; -------------------- SDLイベントループ --------------------
(defun main (&key frame-limit (sync-to-local-time *sync-to-local-time*)
                  (present-mode *present-mode*))
  (let* ((startup-time (get-universal-time))
         ;; 時刻同期時は直前の10分のRLE、同期無効時は従来の初期RLEを読む。
         (source-info (if sync-to-local-time
                          (clock-snapshot-info startup-time)
                          (list (ensure-clock-rle) 0 startup-time)))
         (source-path (first source-info))
         (initial-generation (second source-info))
         (initial-sync-time (third source-info)))
    (multiple-value-bind (initial-words population pattern-width pattern-height
                          initial-active-tiles)
        (parse-rle-into-grid source-path)
      (format t "~&Clock loaded from ~A: ~Dx~D, ~:D live cells, generation ~:D, grid ~Dx~D (~,1F MiB x 2).~%"
              (file-namestring source-path) pattern-width pattern-height population
              initial-generation +grid-width+ +grid-height+
              (/ (* +word-count+ 4) 1048576.0))
    (let ((shader (uiop:read-file-string
                   (merge-pathnames "shader.wgsl" *source-directory*))))
      (sdl2:with-init (:video)
        ;; with-windowを抜けるとSDLウィンドウは自動的に片付けられる。
        (sdl2:with-window (window :title *window-title*
                                  :w *window-width* :h *window-height*
                                  :x *window-x* :y *window-y*
                                  ;; 同期中の未完成な時計を見せないよう、最初は隠す。
                                  ;; ローカル時刻まで進み終えたときだけSDL_ShowWindowで開く。
                                  :flags '(:hidden :resizable))
          ;; GPU資源は初めNULLにしておく。途中で失敗しても、作成済みのものだけ解放できる。
          (let ((instance (cffi:null-pointer))
                (surface (cffi:null-pointer))
                (device (cffi:null-pointer))
                (queue (cffi:null-pointer))
                (state (cffi:null-pointer))
                (configured nil)
                ;; cameraの単位はセル座標。centerは画面中央に見せたい盤面位置。
                (width *window-width*) (height *window-height*)
                (center-x (/ +grid-width+ 2.0f0))
                (center-y (/ +grid-height+ 2.0f0))
                (zoom-level +initial-zoom-level+)
                (zoom (zoom-for-level +initial-zoom-level+))
                (mouse-x 0.0f0) (mouse-y 0.0f0) (dragging nil)
                ;; single-stepはRightキーで予約された「あと何世代だけ進めるか」。
                (paused nil) (single-step 0)
                (observed-sparse-mode *sparse-mode*)
                ;; accumulatorには「進める時刻になったが、まだ実行していない世代数」を貯める。
                (speed (if sync-to-local-time
                           (float +clock-generations-per-second+ 1.0d0)
                           *initial-speed*))
                (generation initial-generation) (accumulator 0.0d0)
                ;; 同期の基準時刻と目標世代。日付をまたいでも目標が0へ戻らないよう、
                ;; 「起動時の世代 + それからの経過秒数」として数える。
                (syncing sync-to-local-time)
                (sync-origin-time initial-sync-time)
                (sync-origin-generation initial-generation)
                (last-sync-report 0)
                ;; 起動同期が終わったときに「次は5分後」と設定する。
                (next-clock-correction-time nil)
                (last-time (get-internal-real-time))
                (stats-start (get-internal-real-time))
                (stats-generation 0) (stats-frame-count 0)
                (measured-speed 0.0d0) (measured-fps 0.0d0)
                (last-title 0) (frames 0))
            (unwind-protect
                 (progn
                   ;;; -------------------- 2. GPUを初期化する --------------------
                   ;; Instance → Surface → Device/Pipeline → Queueの順で用意する。
                   (setf instance (require-handle (%create-instance 1) "Create instance")
                         surface (require-handle
                                  (init-surface instance (autowrap:ptr window))
                                  "Create surface"))
                   (cffi:with-foreign-object (out-state :pointer)
                     ;; out-stateは、C関数がLifeStateのポインターを書き戻すための小さな箱。
                     (setf (cffi:mem-ref out-state :pointer) (cffi:null-pointer))
                     (cffi:with-pointer-to-vector-data (initial-pointer initial-words)
                       (if (zerop (length initial-active-tiles))
                           (setf device
                                 (%init-life instance surface shader initial-pointer
                                             +word-count+ (cffi:null-pointer) 0
                                             out-state))
                           (cffi:with-pointer-to-vector-data
                               (tiles-pointer initial-active-tiles)
                             (setf device
                                   (%init-life instance surface shader initial-pointer
                                               +word-count+ tiles-pointer
                                               (length initial-active-tiles)
                                               out-state)))))
                     (setf state (cffi:mem-ref out-state :pointer)))
                   (require-handle device "Create Life pipelines")
                   (require-handle state "Create Life state")
                   (setf queue (require-handle (get-queue device) "Get queue"))
                   (setf present-mode
                         (configure-surface device surface state width height
                                            present-mode))
                   (setf configured t
                         last-time (get-internal-real-time)
                         stats-start last-time)
                   ;; 同期しない起動方法は、テストや初期盤面の確認に使う。
                   (unless syncing (%show-window (autowrap:ptr window)))

                   ;;; -------------------- 3. 入力と毎フレーム処理 --------------------
                   ;; poll方式は、イベントがなくても:idleを繰り返す。そこで計算と描画を行う。
                   (sdl2:with-event-loop (:method :poll)
                     (:quit () t)
                     (:windowevent (:event event :data1 d1 :data2 d2)
                       ;; リサイズ後の幅・高さをSurfaceへ伝え、描画先を作り直す。
                       (when (or (= event sdl2-ffi:+sdl-windowevent-resized+)
                                 (= event sdl2-ffi:+sdl-windowevent-size-changed+))
                         (setf width d1 height d2)
                         (when (and (plusp d1) (plusp d2))
                           (setf present-mode
                                 (configure-surface device surface state d1 d2
                                                    present-mode)))))
                     (:mousemotion (:x x :y y :xrel xrel :yrel yrel)
                       (setf mouse-x (float x 1.0f0) mouse-y (float y 1.0f0))
                       (when dragging
                         ;; マウスが右へ動いたら、カメラを左へ動かすと「盤面をつかむ」動きになる。
                         (decf center-x (* (float xrel 1.0f0) zoom))
                         (decf center-y (* (float yrel 1.0f0) zoom))))
                     (:mousebuttondown (:button button)
                       (when (= button sdl2-ffi:+sdl-button-left+)
                         (setf dragging t)))
                     (:mousebuttonup (:button button)
                       (when (= button sdl2-ffi:+sdl-button-left+)
                         (setf dragging nil)))
                     (:mousewheel (:y wheel-y)
                       (when (and (plusp width) (plusp height) (/= wheel-y 0))
                         ;; zoom前にカーソル下の盤面座標を記録し、zoom後も同じ場所が
                         ;; カーソル下に来るようcenterを逆算する。
                         (let* ((world-x (+ center-x (* (- mouse-x (/ width 2.0f0)) zoom)))
                                (world-y (+ center-y (* (- mouse-y (/ height 2.0f0)) zoom)))
                                (new-level
                                  (min +max-zoom-level+
                                       (max +min-zoom-level+
                                            (+ zoom-level
                                               (if (plusp wheel-y) -1 1)))))
                                (new-zoom (zoom-for-level new-level)))
                           (setf center-x (- world-x (* (- mouse-x (/ width 2.0f0)) new-zoom))
                                 center-y (- world-y (* (- mouse-y (/ height 2.0f0)) new-zoom))
                                 zoom-level new-level
                                 zoom new-zoom))))
                     (:keydown (:keysym keysym :repeat repeat)
                       ;; 押しっぱなしの自動リピートは無視し、1回押すごとに1段だけ操作する。
                       (when (zerop repeat)
                         (cond
                           ((scancode-is keysym :scancode-escape) (sdl2:push-event :quit))
                           ((scancode-is keysym :scancode-space) (setf paused (not paused)))
                           ((scancode-is keysym :scancode-right)
                            (setf paused t) (incf single-step))
                           ((scancode-is keysym :scancode-up)
                            (setf speed (next-speed-level speed)))
                           ((scancode-is keysym :scancode-down)
                            (setf speed (previous-speed-level speed)))
                           ((scancode-is keysym :scancode-v)
                            (setf present-mode
                                  (configure-surface
                                   device surface state width height
                                   (if (eq present-mode :fifo)
                                       :immediate
                                       :fifo))
                                  ;; 切替前後の値を同じ測定区間へ混ぜない。
                                  stats-start (get-internal-real-time)
                                  stats-generation generation
                                  stats-frame-count 0
                                  measured-speed 0.0d0
                                  measured-fps 0.0d0))
                           ((scancode-is keysym :scancode-f)
                            (when (and (plusp width) (plusp height))
                              (let ((new-level (full-view-zoom-level width height)))
                                (setf center-x (/ +grid-width+ 2.0f0)
                                      center-y (/ +grid-height+ 2.0f0)
                                      zoom-level new-level
                                      zoom (zoom-for-level new-level)))))
                           ((scancode-is keysym :scancode-r)
                            (reset-gpu-grid queue state initial-words
                                            initial-active-tiles)
                            (setf generation initial-generation accumulator 0.0d0 paused t
                                  stats-start (get-internal-real-time)
                                  stats-generation initial-generation stats-frame-count 0
                                  measured-speed 0.0d0
                                  measured-fps 0.0d0)))))
                     (:idle ()
                       ;; now-last-timeが前フレームから経過した秒数。
                       ;; 長い停止後に大量計算しないよう、1回分は最大0.25秒に丸める。
                       (let* ((now (get-internal-real-time))
                              ;; この値は、この:idle処理が始まった時点で同期中かを覚える。
                              ;; 同期完了と同じ回に通常時間も足すと、経過時間を2回数えてしまう。
                              (syncing-at-frame-start syncing)
                              (raw-delta
                                (/ (- now last-time)
                                   (float internal-time-units-per-second 1.0d0)))
                              (delta (min 0.25d0 raw-delta)))
                         (setf last-time now)
                         (unless (eql observed-sparse-mode *sparse-mode*)
                           ;; REPLからモードが変わった場合、2モードの測定値を混ぜない。
                           (setf observed-sparse-mode *sparse-mode*
                                 stats-start now
                                 stats-generation generation
                                 stats-frame-count 0
                                 measured-speed 0.0d0
                                 measured-fps 0.0d0))
                         ;; 同期中は画面を借りず、世代更新だけをGPUへ頼む。
                         ;; SPARSEは生きている周辺だけ計算するので、同期に最も向いている。
                         (when syncing
                           (let* ((wall-now (get-universal-time))
                                  (target (clock-target-generation
                                           wall-now sync-origin-time
                                           sync-origin-generation))
                                  (remaining (- target generation)))
                             (if (plusp remaining)
                                 (let ((batch (min +sync-steps-per-batch+ remaining)))
                                   (when (minusp
                                          (advance-life device queue state batch 1))
                                     (error "Local-time synchronization compute failed"))
                                   (incf generation batch)
                                   ;; 進捗は1秒に1回だけ表示し、ターミナルを流しすぎない。
                                   (when (>= (- now last-sync-report)
                                             internal-time-units-per-second)
                                     (setf last-sync-report now)
                                     (let ((completed (- generation sync-origin-generation))
                                           (total (- target sync-origin-generation)))
                                       (format t "~&Synchronizing 10-minute snapshot: ~:D / ~:D generations (~,1F%)~%"
                                             completed total
                                             (if (plusp total)
                                                 (* 100.0d0 (/ completed total))
                                                 100.0d0))
                                       (finish-output))))
                                 (progn
                                   ;; ここまで画面は一度もpresentしていない。
                                   ;; 時刻が合った盤面ができてから、初めてウィンドウを見せる。
                                   (setf syncing nil
                                         accumulator 0.0d0
                                         last-time now
                                         stats-start now
                                         stats-generation generation
                                         stats-frame-count 0
                                         next-clock-correction-time
                                           (+ wall-now +clock-correction-seconds+))
                                   (%show-window (autowrap:ptr window))
                                   (format t "~&Local-time synchronization complete at generation ~:D.~%"
                                           generation)
                                   (finish-output)))))
                         ;; 例: 0.02秒 × 192 gen/s = 3.84世代をaccumulatorへ追加する。
                         ;; 同期中はadvance-lifeが直接進めるので、ここへは追加しない。
                         (unless (or paused syncing-at-frame-start)
                           (incf accumulator (* delta speed)))
                         ;; 192 gen/sで動いている時計は、5分ごとにOS時刻と比べ直す。
                         ;; 遅れは正の値、進みすぎは負の値としてaccumulatorへ入れる。
                         ;; 負なら自動更新を休み、実時間が追いつくのを待つ。
                         (when (and sync-to-local-time
                                    (not syncing)
                                    next-clock-correction-time
                                    (= speed (float +clock-generations-per-second+
                                                    1.0d0)))
                           (let ((wall-now (get-universal-time)))
                             (when (>= wall-now next-clock-correction-time)
                               (let* ((target (local-generation-at wall-now))
                                      (difference (clock-generation-difference
                                                   target generation)))
                                 (setf accumulator (float difference 1.0d0)
                                       next-clock-correction-time
                                         (+ wall-now +clock-correction-seconds+))
                                 (format t "~&Clock correction: ~:[ahead~;behind/on time~] by ~:D generations.~%"
                                         (not (minusp difference))
                                         (abs difference))
                                 (finish-output)))))
                         (let* ((automatic (if (or paused syncing-at-frame-start)
                                               0
                                               ;; accumulatorが負なら、時計が進みすぎている。
                                               ;; 0世代だけ進めて、OS時刻が追いつくのを待つ。
                                               (max 0 (floor accumulator))))
                                (wanted (+ automatic single-step))
                                (steps (min +max-steps-per-frame+ wanted))
                                (automatic-used (min automatic steps))
                                (single-used (min single-step (- steps automatic-used))))
                           ;; 最小化中は幅や高さが0になる場合があるのでGPU描画を休む。
                           (when (and (not syncing) (plusp width) (plusp height))
                             (let ((result
                                     (draw-life device queue surface state
                                                (float width 1.0f0) (float height 1.0f0)
                                                (float center-x 1.0f0) (float center-y 1.0f0)
                                                (float zoom 1.0f0) steps
                                                (if *sparse-mode* 1 0))))
                               (when (minusp result) (error "Compute/draw frame failed"))
                               (when (zerop result)
                                 ;; C側が成功した世代だけ、予約数と表示上の世代数から差し引く。
                                 (decf accumulator automatic-used)
                                 (decf single-step single-used)
                                 (incf generation steps)
                                 (incf frames)
                                 (incf stats-frame-count)))))
                         ;; 約1秒ごとに、実際に進んだ世代数と描いたフレーム数を割って測定する。
                         (let ((stats-elapsed
                                 (/ (- now stats-start)
                                    (float internal-time-units-per-second 1.0d0))))
                           (when (>= stats-elapsed 1.0d0)
                             (setf measured-speed
                                   (/ (- generation stats-generation) stats-elapsed)
                                   measured-fps (/ stats-frame-count stats-elapsed)
                                   stats-start now
                                   stats-generation generation
                                   stats-frame-count 0)))
                         ;; タイトル更新は毎フレームではなく毎秒4回までにして無駄を減らす。
                         (when (> (- now last-title)
                                  (/ internal-time-units-per-second 4))
                           (setf last-title now)
                           (%set-window-title
                            (autowrap:ptr window)
                            (format nil "Life 20000² | ~:[DENSE~;SPARSE~] | ~A | gen ~:D | ~:[RUN~;PAUSE~] | target ~A gen/s | actual ~,1F gen/s | ~,1F fps | zoom ~,1F cell/px"
                                    *sparse-mode*
                                    (if (eq present-mode :immediate)
                                        "IMMEDIATE"
                                        "FIFO")
                                    generation paused
                                    (if (= speed 192.0d0)
                                        "*192*"
                                        (format nil "~,1F" speed))
                                    measured-speed
                                    measured-fps zoom)))
                         (when (and frame-limit (>= frames frame-limit))
                           (sdl2:push-event :quit))
                         (sdl2:delay 1))))

              ;;; -------------------- 4. 終了時の後片付け --------------------
              ;; unwind-protectの後半なので、途中でエラーが起きてもここは実行される。
              ;; 作成とおおむね逆の順番でGPU資源を解放する。
              (unless (cffi:null-pointer-p state) (release-life state))
              (when configured (unconfigure-surface surface))
              (unless (cffi:null-pointer-p queue) (release-queue queue))
              (unless (cffi:null-pointer-p device) (release-device device))
              (unless (cffi:null-pointer-p surface) (release-surface surface))
              (unless (cffi:null-pointer-p instance) (release-instance instance)))))))))))

(defun run-main-thread ()
  ;; SDLのイベントループを専用スレッドで動かす。REPLは操作可能なまま残る。
  (when (and *main-thread* (bt:thread-alive-p *main-thread*))
    (error "Life simulator is already running"))
  (setf *main-thread* (bt:make-thread #'main :name "SDL2-WGPU-Life")))

(when *auto-start*
  ;; LIFEGAME_NO_AUTOSTART=1ならロードだけ行い、手動でmainを呼べる。
  (run-main-thread))
