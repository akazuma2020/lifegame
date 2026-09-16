(ql:quickload '(:cffi :sdl2 :bordeaux-threads))

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
  (uiop:pathname-directory-pathname
   (or *load-truename* *compile-file-truename*)))

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
  "https://gist.githubusercontent.com/anonymous/f3413564b1fa9c69f2bad4b0400b8090/raw/f5c77c999a8e11f0ec6ba504d383774eb3b88e5c/Conway%2520life%2520clock%2520PM%2520only")
(defparameter *clock-path* (merge-pathnames "clock.rle" *source-directory*))
(defparameter *clock-snapshot-directory*
  (merge-pathnames "clock-snapshot/" *source-directory*))

(defparameter *video-driver*
  (or (uiop:getenv "LIFEGAME_SDL_VIDEODRIVER") "x11"))
(setf (uiop:getenv "SDL_VIDEODRIVER") *video-driver*
      (uiop:getenv "RUST_BACKTRACE") "full")

;;; -------------------- C bridgeとの接続 --------------------
(cffi:load-foreign-library "/usr/lib/wsl/lib/libd3d12core.so")
(cffi:load-foreign-library "/usr/lib/wsl/lib/libd3d12.so")
(cffi:load-foreign-library "libwgpu_native.so")
(cffi:load-foreign-library
 (merge-pathnames "bridge/bridge.so" *source-directory*))

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
  (device :pointer) (surface :pointer) (width :uint32) (height :uint32))
(cffi:defcfun ("WGPU_AdvanceLife" advance-life) :int
  (device :pointer) (queue :pointer) (state :pointer)
  (steps :uint32) (sparse-mode :uint32))
(cffi:defcfun ("WGPU_DrawLife" draw-life) :int
  (device :pointer) (queue :pointer) (surface :pointer) (state :pointer)
  (width :float) (height :float) (center-x :float) (center-y :float)
  (zoom :float) (steps :uint32) (sparse-mode :uint32))
(cffi:defcfun ("WGPU_ResetLife" %reset-life) :void
  (queue :pointer) (state :pointer) (initial-words :pointer)
  (word-count :uint64)
  (initial-active-tiles :pointer) (initial-tile-count :uint32))
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

;;; -------------------- 実行時設定 --------------------
(defparameter *window-title* "WGPU Life 20000 x 20000")
(defparameter *window-width* 1200)
(defparameter *window-height* 800)
(defparameter *window-x* 80)
(defparameter *window-y* 80)
(defparameter *speed-levels*
  #(1.0d0 2.0d0 3.0d0 4.0d0 5.0d0
    10.0d0 20.0d0 30.0d0 40.0d0 50.0d0
    60.0d0 70.0d0 80.0d0 90.0d0 100.0d0
    110.0d0 120.0d0 130.0d0 140.0d0 150.0d0
    160.0d0 170.0d0 180.0d0 190.0d0 192.0d0 200.0d0
    210.0d0 220.0d0 230.0d0 240.0d0 250.0d0
    260.0d0 270.0d0 280.0d0 290.0d0 300.0d0))
(defparameter *initial-speed* 50.0d0)
(defparameter *sparse-mode* t)
(defparameter *sync-to-local-time* t)
(defvar *main-thread* nil)
(defparameter *auto-start*
  (not (string= (or (uiop:getenv "LIFEGAME_NO_AUTOSTART") "") "1")))

(defun require-handle (handle description)
  (when (cffi:null-pointer-p handle)
    (error "~A failed" description))
  handle)

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
  (cffi:with-pointer-to-vector-data (words-pointer words)
    (if (zerop (length active-tiles))
        (%reset-life queue state words-pointer +word-count+
                     (cffi:null-pointer) 0)
        (cffi:with-pointer-to-vector-data (tiles-pointer active-tiles)
          (%reset-life queue state words-pointer +word-count+
                       tiles-pointer (length active-tiles))))))

(defun scancode-is (keysym code)
  (sdl2:scancode= (sdl2:scancode-value keysym) code))

(defun next-speed-level (speed)
  (or (find-if (lambda (level) (> level speed)) *speed-levels*)
      (aref *speed-levels* (1- (length *speed-levels*)))))

(defun previous-speed-level (speed)
  (or (find-if (lambda (level) (< level speed))
               *speed-levels* :from-end t)
      (aref *speed-levels* 0)))

;;; -------------------- SDLイベントループ --------------------
(defun main (&key frame-limit (sync-to-local-time *sync-to-local-time*))
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
                   (when (minusp (update-surface device surface width height))
                     (error "Configure surface failed"))
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
                           (when (minusp (update-surface device surface d1 d2))
                             (error "Resize surface failed")))))
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
                            (format nil "Life 20000² | ~:[DENSE~;SPARSE~] | gen ~:D | ~:[RUN~;PAUSE~] | target ~A gen/s | actual ~,1F gen/s | ~,1F fps | zoom ~,1F cell/px"
                                    *sparse-mode* generation paused
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
