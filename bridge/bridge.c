/*
 * SDL2 + wgpu-native bridge for the 20000 x 20000 Life simulator.
 *
 * このファイルは、LispとGPUの間にある「通訳」です。
 * Lispは操作や時間を管理し、このCコードはWebGPUへ具体的な命令を渡します。
 * セルの計算式そのものはshader.wgslにあります。
 *
 * 主な流れ:
 *   Lisp → WGPU_InitLife  : GPUと必要なバッファを準備
 *   Lisp → WGPU_AdvanceLife: 画面を出さずに世代だけ進める
 *   Lisp → WGPU_DrawLife  : 世代更新と1画面の描画を依頼
 *   C    → shader.wgsl    : compute/render pipelineを通してGPUで実行
 *
 * Cでは、WGPUBufferなどの「handle」がGPU上の物を指します。使い終えたhandleは
 * Release関数で返さないと、GPUメモリが使われたままになります。
 */

#include <webgpu/wgpu.h>

/*
 * SDL2の公式Windows開発パッケージはinclude直下にSDL.hを置く。一方、
 * Debian/UbuntuはSDL2/SDL.hとしてインストールするため、OSごとに分ける。
 */
#if defined(_WIN32)
#ifndef SDL_MAIN_HANDLED
#define SDL_MAIN_HANDLED
#endif
/*
 * MSYS2のSDL2 headerはMinGW向けSDL_config.hを同梱し、MSVCには存在しない
 * strings.hとGCC atomic builtinsを有効にしている。SDL headerより先にconfigを
 * 一度読み、MSVCで非互換な項目だけを解除する。SDL2.dllのABIには影響しない。
 */
#if defined(_MSC_VER)
#include <SDL_config.h>
#ifdef HAVE_STRINGS_H
#undef HAVE_STRINGS_H
#endif
#ifdef HAVE_GCC_ATOMICS
#undef HAVE_GCC_ATOMICS
#endif
#ifdef HAVE_GCC_SYNC_LOCK_TEST_AND_SET
#undef HAVE_GCC_SYNC_LOCK_TEST_AND_SET
#endif
#endif
#include <SDL.h>
#include <SDL_syswm.h>
#else
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Lisp/CFFIから呼ぶ9関数をWindows DLLのexport tableへ公開する。 */
#if defined(_WIN32) && !defined(LIFE_SHADER_VALIDATE)
#define LIFE_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define LIFE_API __attribute__((visibility("default")))
#else
#define LIFE_API
#endif

/* -------------------- Lispから使う汎用command batch -------------------- */
/*
 * Lispからwgpu-nativeを1命令ずつ呼ぶと、LispとCを何度も往復する。
 * そこで、Lispは「やることリスト」だけを作り、Cがまとめて実行する。
 * この仕組みはLife専用ではなく、別のCompute実験でも使える。
 */
typedef enum {
    WGPU_BRIDGE_COMPUTE_CLEAR_BUFFER = 1,
    WGPU_BRIDGE_COMPUTE_BEGIN_PASS = 2,
    WGPU_BRIDGE_COMPUTE_SET_PIPELINE = 3,
    WGPU_BRIDGE_COMPUTE_SET_BIND_GROUP = 4,
    WGPU_BRIDGE_COMPUTE_DISPATCH = 5,
    WGPU_BRIDGE_COMPUTE_DISPATCH_INDIRECT = 6,
    WGPU_BRIDGE_COMPUTE_END_PASS = 7
} WGPUBridgeComputeOpcode;

typedef struct {
    uint32_t opcode;
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint64_t offset;
    uint64_t size;
    void *object;
    void *argument;
} WGPUBridgeComputeCommand;

LIFE_API int WGPU_EncodeComputeCommands(
    WGPUCommandEncoder encoder,
    const WGPUBridgeComputeCommand *commands,
    uint32_t command_count)
{
    if (!encoder || (!commands && command_count)) return -1;

    WGPUComputePassEncoder pass = NULL;
    for (uint32_t i = 0; i < command_count; i++) {
        const WGPUBridgeComputeCommand *command = &commands[i];
        switch ((WGPUBridgeComputeOpcode)command->opcode) {
        case WGPU_BRIDGE_COMPUTE_CLEAR_BUFFER:
            if (pass || !command->object || !command->size) goto fail;
            wgpuCommandEncoderClearBuffer(
                encoder, (WGPUBuffer)command->object,
                command->offset, command->size);
            break;
        case WGPU_BRIDGE_COMPUTE_BEGIN_PASS:
            if (pass) goto fail;
            pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
            if (!pass) goto fail;
            break;
        case WGPU_BRIDGE_COMPUTE_SET_PIPELINE:
            if (!pass || !command->object) goto fail;
            wgpuComputePassEncoderSetPipeline(
                pass, (WGPUComputePipeline)command->object);
            break;
        case WGPU_BRIDGE_COMPUTE_SET_BIND_GROUP:
            if (!pass || !command->object) goto fail;
            wgpuComputePassEncoderSetBindGroup(
                pass, command->x, (WGPUBindGroup)command->object, 0, NULL);
            break;
        case WGPU_BRIDGE_COMPUTE_DISPATCH:
            if (!pass || !command->x || !command->y || !command->z) goto fail;
            wgpuComputePassEncoderDispatchWorkgroups(
                pass, command->x, command->y, command->z);
            break;
        case WGPU_BRIDGE_COMPUTE_DISPATCH_INDIRECT:
            if (!pass || !command->object) goto fail;
            wgpuComputePassEncoderDispatchWorkgroupsIndirect(
                pass, (WGPUBuffer)command->object, command->offset);
            break;
        case WGPU_BRIDGE_COMPUTE_END_PASS:
            if (!pass) goto fail;
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
            pass = NULL;
            break;
        default:
            goto fail;
        }
    }
    if (pass) goto fail;
    return 0;

fail:
    if (pass) wgpuComputePassEncoderRelease(pass);
    return -1;
}

static int submit_encoder(WGPUDevice device, WGPUQueue queue,
                          WGPUCommandEncoder encoder, bool wait)
{
    WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
    if (!command) return -1;

    if (wait) {
        WGPUSubmissionIndex index = wgpuQueueSubmitForIndex(queue, 1, &command);
        wgpuCommandBufferRelease(command);
        return wgpuDevicePoll(device, true, &index) ? 0 : -1;
    }

    wgpuQueueSubmit(queue, 1, &command);
    wgpuCommandBufferRelease(command);
    return 0;
}

LIFE_API int WGPU_RunComputeCommands(
    WGPUDevice device, WGPUQueue queue,
    const WGPUBridgeComputeCommand *commands,
    uint32_t command_count, uint32_t wait)
{
    if (!device || !queue || (!commands && command_count)) return -1;

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    if (!encoder) return -1;

    int result = WGPU_EncodeComputeCommands(encoder, commands, command_count);
    if (result == 0) result = submit_encoder(device, queue, encoder, wait != 0);
    wgpuCommandEncoderRelease(encoder);
    return result;
}

LIFE_API int WGPU_RunComputeRenderFrame(
    WGPUDevice device, WGPUQueue queue, WGPUSurface surface,
    const WGPUBridgeComputeCommand *commands, uint32_t command_count,
    WGPURenderPipeline pipeline, WGPUBindGroup bind_group,
    double clear_r, double clear_g, double clear_b, double clear_a)
{
    if (!device || !queue || !surface || !pipeline || !bind_group) return -1;
    if (!commands && command_count) return -1;

    WGPUSurfaceTexture surface_texture = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(surface, &surface_texture);
    if (surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Timeout ||
        surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Outdated ||
        surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Lost) {
        if (surface_texture.texture) wgpuTextureRelease(surface_texture.texture);
        return 1;
    }
    if (!surface_texture.texture) return -1;

    WGPUTextureView view = wgpuTextureCreateView(surface_texture.texture, NULL);
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    if (!view || !encoder) goto fail;
    if (WGPU_EncodeComputeCommands(encoder, commands, command_count) < 0) goto fail;

    WGPURenderPassColorAttachment color =
        WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    color.view = view;
    color.loadOp = WGPULoadOp_Clear;
    color.storeOp = WGPUStoreOp_Store;
    color.clearValue = (WGPUColor){clear_r, clear_g, clear_b, clear_a};

    WGPURenderPassDescriptor descriptor = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    descriptor.colorAttachmentCount = 1;
    descriptor.colorAttachments = &color;

    WGPURenderPassEncoder pass =
        wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    if (!pass) goto fail;
    wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    if (submit_encoder(device, queue, encoder, false) < 0) goto fail;

    /*
     * SurfaceのTextureを解放する前にpresentする。
     * 先に解放すると、画面へ出すTextureがなくなり黒画面になる。
     */
    wgpuSurfacePresent(surface);

    wgpuCommandEncoderRelease(encoder);
    wgpuTextureViewRelease(view);
    wgpuTextureRelease(surface_texture.texture);
    return 0;

fail:
    if (encoder) wgpuCommandEncoderRelease(encoder);
    if (view) wgpuTextureViewRelease(view);
    wgpuTextureRelease(surface_texture.texture);
    return -1;
}

/* -------------------- 盤面サイズとGPUバッファの大きさ -------------------- */
#define GRID_WIDTH 20000u
#define GRID_HEIGHT 20000u
/* 横32セルをuint32_t 1個へ詰める。+31してから割るのは切り上げの定番式。 */
#define WORDS_PER_ROW ((GRID_WIDTH + 31u) / 32u)
#define WORD_COUNT ((uint64_t)WORDS_PER_ROW * GRID_HEIGHT)
#define GRID_BYTES (WORD_COUNT * sizeof(uint32_t))
#define UNIFORM_BYTES 48u
#define CORE_WORDS 4u
#define CORE_ROWS 128u
#define TILE_COLUMNS ((WORDS_PER_ROW + CORE_WORDS - 1u) / CORE_WORDS)
#define TILE_ROWS ((GRID_HEIGHT + CORE_ROWS - 1u) / CORE_ROWS)
#define TILE_COUNT (TILE_COLUMNS * TILE_ROWS)
#define TILE_BUFFER_BYTES ((uint64_t)TILE_COUNT * sizeof(uint32_t))
#define INDIRECT_BYTES (3u * sizeof(uint32_t))

/*
 * LifeStateは、実行中ずっと保持するGPU資源の一覧表。
 * cells[0]とcells[1]は「現在」と「次世代」を交互に担当する2枚の盤面である。
 * currentが0ならcells[0]が入力、1回進めるとcurrent=1になり役割が入れ替わる。
 * この方式をdouble buffering（ダブルバッファ）という。
 *
 * tile_flags/active_tiles/indirect_argsも2組ある。盤面と同じ番号にそろえることで、
 * 「この盤面を次に計算するときの候補タイル」を取り違えない。
 */
typedef struct {
    WGPURenderPipeline render_pipeline;
    WGPUComputePipeline step_one_pipeline;
    WGPUComputePipeline step_eight_pipeline;
    WGPUComputePipeline clear_pipeline;
    WGPUBuffer uniforms;
    WGPUBuffer cells[2];
    WGPUBuffer tile_flags[2];
    WGPUBuffer active_tiles[2];
    WGPUBuffer indirect_args[2];
    WGPUBindGroup render_groups[2];
    WGPUBindGroup step_one_groups[2];
    WGPUBindGroup step_eight_groups[2];
    WGPUBindGroup clear_groups[2];
    uint32_t current;
    WGPUPresentMode present_mode;
    bool immediate_supported;
} LifeState;

/*
 * AdapterはGPU候補、Deviceは選ばれたGPUへ命令を出す入口。取得は非同期で行われる。
 * MSVCはC11 stdatomic.hの対応状況が版によって異なる。既に依存しているSDL2の
 * atomicを使えば、GCC/MSVCの両方でcallbackと待機側を安全に同期できる。
 */
typedef SDL_atomic_t LifeAtomicBool;
typedef struct { WGPUAdapter adapter; LifeAtomicBool done; } AdapterRequest;
typedef struct { WGPUDevice device; LifeAtomicBool done; } DeviceRequest;

static bool life_atomic_load(LifeAtomicBool *value)
{
    return SDL_AtomicGet(value) != 0;
}

static void life_atomic_store(LifeAtomicBool *value, bool state)
{
    SDL_AtomicSet(value, state ? 1 : 0);
}

static WGPUStringView sv(const char *text)
{
    /* WebGPUの文字列は、先頭アドレスだけでなく長さも一緒に渡す。 */
    WGPUStringView result = { .data = text, .length = strlen(text) };
    return result;
}

static void adapter_callback(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                             WGPUStringView message, void *userdata1, void *userdata2)
{
    /* GPU探しが終わったときwgpu-nativeから呼ばれる。doneを最後にtrueにする。 */
    AdapterRequest *request = userdata1;
    (void)message; (void)userdata2;
    if (status == WGPURequestAdapterStatus_Success) request->adapter = adapter;
    life_atomic_store(&request->done, true);
}

static void device_callback(WGPURequestDeviceStatus status, WGPUDevice device,
                            WGPUStringView message, void *userdata1, void *userdata2)
{
    /* Device作成完了の通知。SDL atomicなら別スレッドから安全に読み書きできる。 */
    DeviceRequest *request = userdata1;
    (void)message; (void)userdata2;
    if (status == WGPURequestDeviceStatus_Success) request->device = device;
    life_atomic_store(&request->done, true);
}

LIFE_API WGPUInstance WGPU_CreateInstance(int allow_noncompliant)
{
    /* InstanceはWebGPU利用全体の出発点。WSLのDozenも候補に含められる設定を足す。 */
    WGPUInstanceExtras extras = {0};
    extras.chain.sType = (WGPUSType)WGPUSType_InstanceExtras;
    extras.flags = WGPUInstanceFlag_Default;
    if (allow_noncompliant)
        extras.flags |= WGPUInstanceFlag_AllowUnderlyingNoncompliantAdapter;
    WGPUInstanceDescriptor descriptor = { .nextInChain = &extras.chain };
    return wgpuCreateInstance(&descriptor);
}

LIFE_API WGPUSurface WGPU_InitSurface(WGPUInstance instance, SDL_Window *window)
{
    /*
     * Surfaceは「GPUの絵を表示するウィンドウ」との接続口。
     * SDLからOS固有情報を取り出し、Win32、X11、Waylandのいずれかに合う
     * 部品をWebGPUへ渡す。使えないunion memberはプリプロセッサで除外する。
     */
    if (!instance || !window) return NULL;
    SDL_SysWMinfo wm;
    SDL_VERSION(&wm.version);
    if (!SDL_GetWindowWMInfo(window, &wm)) return NULL;

#if defined(SDL_VIDEO_DRIVER_WINDOWS)
    if (wm.subsystem == SDL_SYSWM_WINDOWS) {
        WGPUSurfaceSourceWindowsHWND source = {
            .chain = { .sType = WGPUSType_SurfaceSourceWindowsHWND },
            .hinstance = wm.info.win.hinstance,
            .hwnd = wm.info.win.window
        };
        WGPUSurfaceDescriptor descriptor = { .nextInChain = &source.chain };
        return wgpuInstanceCreateSurface(instance, &descriptor);
    }
#endif
#if defined(SDL_VIDEO_DRIVER_X11)
    if (wm.subsystem == SDL_SYSWM_X11) {
        WGPUSurfaceSourceXlibWindow source = {
            .chain = { .sType = WGPUSType_SurfaceSourceXlibWindow },
            .display = wm.info.x11.display,
            .window = (uint64_t)wm.info.x11.window
        };
        WGPUSurfaceDescriptor descriptor = { .nextInChain = &source.chain };
        return wgpuInstanceCreateSurface(instance, &descriptor);
    }
#endif
#if defined(SDL_VIDEO_DRIVER_WAYLAND)
    if (wm.subsystem == SDL_SYSWM_WAYLAND) {
        WGPUSurfaceSourceWaylandSurface source = {
            .chain = { .sType = WGPUSType_SurfaceSourceWaylandSurface },
            .display = wm.info.wl.display,
            .surface = wm.info.wl.surface
        };
        WGPUSurfaceDescriptor descriptor = { .nextInChain = &source.chain };
        return wgpuInstanceCreateSurface(instance, &descriptor);
    }
#endif
    return NULL;
}

static WGPUShaderModule make_shader(WGPUDevice device, const char *code)
{
    /* shader.wgslの文字列を、GPUがpipeline作成に使えるShaderModuleへ変換する。 */
    WGPUShaderSourceWGSL wgsl = {
        .chain = { .sType = WGPUSType_ShaderSourceWGSL },
        .code = { .data = code, .length = strlen(code) }
    };
    WGPUShaderModuleDescriptor descriptor = { .nextInChain = &wgsl.chain };
    return wgpuDeviceCreateShaderModule(device, &descriptor);
}

static WGPUComputePipeline make_compute_pipeline(WGPUDevice device,
                                                 WGPUShaderModule module,
                                                 const char *entry_point)
{
    /* entry_pointはWGSL内で開始する関数名（step_one等）。 */
    WGPUComputePipelineDescriptor descriptor = {0};
    descriptor.compute.module = module;
    descriptor.compute.entryPoint = sv(entry_point);
    return wgpuDeviceCreateComputePipeline(device, &descriptor);
}

static WGPURenderPipeline make_render_pipeline(WGPUDevice device, WGPUShaderModule module)
{
    /*
     * 描画pipelineは、頂点処理vs_mainと色処理fs_mainを1本につなぐ。
     * BGRA8Unormは青・緑・赤・透明度を各8ビットで持つ一般的な画面形式。
     */
    WGPUColorTargetState target = {
        .format = WGPUTextureFormat_BGRA8Unorm,
        .writeMask = WGPUColorWriteMask_All
    };
    WGPUFragmentState fragment = {
        .module = module,
        .entryPoint = { .data = "fs_main", .length = 7 },
        .targetCount = 1,
        .targets = &target
    };
    WGPURenderPipelineDescriptor descriptor = {0};
    descriptor.vertex.module = module;
    descriptor.vertex.entryPoint = sv("vs_main");
    descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    descriptor.multisample.count = 1;
    descriptor.multisample.mask = 0xffffffffu;
    descriptor.fragment = &fragment;
    return wgpuDeviceCreateRenderPipeline(device, &descriptor);
}

static WGPUBindGroup make_step_group(WGPUDevice device,
                                     WGPUBindGroupLayout layout,
                                     WGPUBuffer uniforms,
                                     WGPUBuffer input,
                                     WGPUBuffer output,
                                     WGPUBuffer active_tiles,
                                     WGPUBuffer next_flags,
                                     WGPUBuffer next_active_tiles,
                                     WGPUBuffer next_indirect)
{
    /*
     * BindGroupは、WGSLの@binding(0)..(6)へ実際のバッファを差し込む配線表。
     * inputとoutputを逆にしたBindGroupを2個作れば、毎世代作り直さず交互に使える。
     */
    WGPUBindGroupEntry entries[7] = {0};
    entries[0].binding = 0; entries[0].buffer = uniforms; entries[0].size = UNIFORM_BYTES;
    entries[1].binding = 1; entries[1].buffer = input; entries[1].size = GRID_BYTES;
    entries[2].binding = 2; entries[2].buffer = output; entries[2].size = GRID_BYTES;
    entries[3].binding = 3; entries[3].buffer = active_tiles; entries[3].size = TILE_BUFFER_BYTES;
    entries[4].binding = 4; entries[4].buffer = next_flags; entries[4].size = TILE_BUFFER_BYTES;
    entries[5].binding = 5; entries[5].buffer = next_active_tiles; entries[5].size = TILE_BUFFER_BYTES;
    entries[6].binding = 6; entries[6].buffer = next_indirect; entries[6].size = INDIRECT_BYTES;
    WGPUBindGroupDescriptor descriptor = {
        .layout = layout, .entryCount = 7, .entries = entries
    };
    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

static WGPUBindGroup make_clear_group(WGPUDevice device,
                                      WGPUBindGroupLayout layout,
                                      WGPUBuffer uniforms,
                                      WGPUBuffer cells,
                                      WGPUBuffer active_tiles)
{
    /* clear_tilesが必要とするuniform、消去先cells、候補一覧だけを配線する。 */
    WGPUBindGroupEntry entries[3] = {0};
    entries[0].binding = 0; entries[0].buffer = uniforms; entries[0].size = UNIFORM_BYTES;
    entries[1].binding = 2; entries[1].buffer = cells; entries[1].size = GRID_BYTES;
    entries[2].binding = 3; entries[2].buffer = active_tiles; entries[2].size = TILE_BUFFER_BYTES;
    WGPUBindGroupDescriptor descriptor = {
        .layout = layout, .entryCount = 3, .entries = entries
    };
    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

static void upload_initial_activity(WGPUQueue queue, LifeState *state,
                                    const uint32_t *tiles, uint32_t count)
{
    /*
     * indirect[0]は実行するworkgroup数。y,zは1固定。
     * 初回はどちらの盤面から開始しても同じ候補になるよう、2組とも初期化する。
     */
    const uint32_t indirect[3] = { count, 1u, 1u };
    for (unsigned i = 0; i < 2; i++) {
        if (count)
            wgpuQueueWriteBuffer(queue, state->active_tiles[i], 0,
                                 tiles, (uint64_t)count * sizeof(*tiles));
        wgpuQueueWriteBuffer(queue, state->indirect_args[i], 0,
                             indirect, sizeof(indirect));
    }
}

static WGPUBindGroup make_render_group(WGPUDevice device,
                                       WGPUBindGroupLayout layout,
                                       WGPUBuffer uniforms,
                                       WGPUBuffer input)
{
    /* 描画shaderは画面設定と現在のセル面だけを読む。 */
    WGPUBindGroupEntry entries[2] = {0};
    entries[0].binding = 0; entries[0].buffer = uniforms; entries[0].size = UNIFORM_BYTES;
    entries[1].binding = 1; entries[1].buffer = input; entries[1].size = GRID_BYTES;
    WGPUBindGroupDescriptor descriptor = {
        .layout = layout, .entryCount = 2, .entries = entries
    };
    return wgpuDeviceCreateBindGroup(device, &descriptor);
}

LIFE_API void WGPU_ReleaseLife(LifeState *state)
{
    /*
     * 作成したGPU資源をすべて解放する。NULLか確認するため、初期化途中の失敗にも使える。
     * 配線（BindGroup）→材料（Buffer/Pipeline）→CPU上のstateの順で片付ける。
     */
    if (!state) return;
    for (unsigned i = 0; i < 2; i++) {
        if (state->render_groups[i]) wgpuBindGroupRelease(state->render_groups[i]);
        if (state->step_one_groups[i]) wgpuBindGroupRelease(state->step_one_groups[i]);
        if (state->step_eight_groups[i]) wgpuBindGroupRelease(state->step_eight_groups[i]);
        if (state->clear_groups[i]) wgpuBindGroupRelease(state->clear_groups[i]);
        if (state->indirect_args[i]) wgpuBufferRelease(state->indirect_args[i]);
        if (state->active_tiles[i]) wgpuBufferRelease(state->active_tiles[i]);
        if (state->tile_flags[i]) wgpuBufferRelease(state->tile_flags[i]);
        if (state->cells[i]) wgpuBufferRelease(state->cells[i]);
    }
    if (state->uniforms) wgpuBufferRelease(state->uniforms);
    if (state->render_pipeline) wgpuRenderPipelineRelease(state->render_pipeline);
    if (state->step_one_pipeline) wgpuComputePipelineRelease(state->step_one_pipeline);
    if (state->step_eight_pipeline) wgpuComputePipelineRelease(state->step_eight_pipeline);
    if (state->clear_pipeline) wgpuComputePipelineRelease(state->clear_pipeline);
    free(state);
}

LIFE_API WGPUDevice WGPU_InitLife(WGPUInstance instance, WGPUSurface surface,
                                  const char *shader_code,
                                  const uint32_t *initial_words,
                                  uint64_t word_count,
                                  const uint32_t *initial_tiles,
                                  uint32_t initial_tile_count,
                                  LifeState **out_state)
{
    /* -------------------- GPU側のLife実行環境を組み立てる --------------------
     * 戻り値はDevice、out_stateにはそのDeviceで作った全資源の一覧を返す。
     * 最初に引数を厳しく確認し、配列サイズの食い違いによるメモリ破壊を防ぐ。
     */
    if (!instance || !surface || !shader_code || !initial_words || !out_state ||
        word_count != WORD_COUNT || initial_tile_count > TILE_COUNT ||
        (initial_tile_count && !initial_tiles)) return NULL;
    *out_state = NULL;

    /* 1. このSurfaceへ表示できる、高性能なGPU（Adapter）を探す。 */
    AdapterRequest adapter_request = {0};
    WGPURequestAdapterOptions options = {
        .compatibleSurface = surface,
        .powerPreference = WGPUPowerPreference_HighPerformance
    };
    WGPURequestAdapterCallbackInfo adapter_info = {
        .callback = adapter_callback, .userdata1 = &adapter_request,
        .mode = WGPUCallbackMode_AllowSpontaneous
    };
    wgpuInstanceRequestAdapter(instance, &options, adapter_info);
    /* callbackがdoneを立てるまで、SDL_DelayでCPUを休ませながら待つ。 */
    while (!life_atomic_load(&adapter_request.done)) SDL_Delay(1);
    if (!adapter_request.adapter) return NULL;

    /* 選ばれたGPU名をログへ出し、CPUだけで動く遅いsoftware adapterは拒否する。 */
    WGPUAdapterInfo info = {0};
    if (wgpuAdapterGetInfo(adapter_request.adapter, &info) == WGPUStatus_Success) {
        const char *kind = info.adapterType == WGPUAdapterType_CPU ? "CPU" :
            info.adapterType == WGPUAdapterType_IntegratedGPU ? "IntegratedGPU" :
            info.adapterType == WGPUAdapterType_DiscreteGPU ? "DiscreteGPU" : "Unknown";
        fprintf(stderr, "[Life] adapter=%.*s backend=%d type=%s\n",
                (int)info.device.length, info.device.data, info.backendType, kind);
        bool software = info.adapterType == WGPUAdapterType_CPU;
        wgpuAdapterInfoFreeMembers(info);
        if (software) {
            fprintf(stderr, "[Life] CPU adapter refused; hardware GPU is required.\n");
            wgpuAdapterRelease(adapter_request.adapter);
            return NULL;
        }
    }

    /* Surfaceごとに利用可能な表示方式は異なる。FIFOは必須だがImmediateは任意。 */
    bool immediate_supported = false;
    WGPUSurfaceCapabilities surface_capabilities = {0};
    if (wgpuSurfaceGetCapabilities(surface, adapter_request.adapter,
                                   &surface_capabilities) == WGPUStatus_Success) {
        for (size_t i = 0; i < surface_capabilities.presentModeCount; i++) {
            if (surface_capabilities.presentModes[i] == WGPUPresentMode_Immediate) {
                immediate_supported = true;
                break;
            }
        }
        wgpuSurfaceCapabilitiesFreeMembers(surface_capabilities);
    }

    /* 2. Adapterから、実際にpipelineやbufferを作るDeviceを取得する。 */
    DeviceRequest device_request = {0};
    WGPUDeviceDescriptor device_descriptor = {0};
    WGPURequestDeviceCallbackInfo device_info = {
        .callback = device_callback, .userdata1 = &device_request,
        .mode = WGPUCallbackMode_AllowSpontaneous
    };
    wgpuAdapterRequestDevice(adapter_request.adapter, &device_descriptor, device_info);
    while (!life_atomic_load(&device_request.done)) SDL_Delay(1);
    wgpuAdapterRelease(adapter_request.adapter);
    if (!device_request.device) return NULL;

    /* callocは全フィールドを0/NULLにする。失敗時の一括解放を安全にするため重要。 */
    WGPUDevice device = device_request.device;
    LifeState *state = calloc(1, sizeof(*state));
    if (!state) { wgpuDeviceRelease(device); return NULL; }

    /* 3. 1個のWGSLから、3本の計算pipelineと1本の描画pipelineを作る。 */
    WGPUShaderModule shader = make_shader(device, shader_code);
    if (!shader) goto fail;
    state->step_one_pipeline = make_compute_pipeline(device, shader, "step_one");
    state->step_eight_pipeline = make_compute_pipeline(device, shader, "step_eight");
    state->clear_pipeline = make_compute_pipeline(device, shader, "clear_tiles");
    state->render_pipeline = make_render_pipeline(device, shader);
    wgpuShaderModuleRelease(shader);
    if (!state->step_one_pipeline || !state->step_eight_pipeline ||
        !state->clear_pipeline || !state->render_pipeline)
        goto fail;

    /* 4. GPUバッファを作る。usageは「何に使ってよいメモリか」を表す許可証。 */
    WGPUBufferDescriptor uniform_descriptor = {
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
        .size = UNIFORM_BYTES
    };
    state->uniforms = wgpuDeviceCreateBuffer(device, &uniform_descriptor);
    WGPUBufferDescriptor cell_descriptor = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = GRID_BYTES
    };
    /* cellsは各約47.7MiB。入力と出力を交換するので2枚必要。 */
    state->cells[0] = wgpuDeviceCreateBuffer(device, &cell_descriptor);
    state->cells[1] = wgpuDeviceCreateBuffer(device, &cell_descriptor);
    WGPUBufferDescriptor flags_descriptor = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = TILE_BUFFER_BYTES
    };
    WGPUBufferDescriptor tiles_descriptor = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
        .size = TILE_BUFFER_BYTES
    };
    WGPUBufferDescriptor indirect_descriptor = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect |
                 WGPUBufferUsage_CopyDst,
        .size = INDIRECT_BYTES
    };
    /* flagsは重複防止、active_tilesは候補番号、indirect_argsは候補数を持つ。 */
    for (unsigned i = 0; i < 2; i++) {
        state->tile_flags[i] = wgpuDeviceCreateBuffer(device, &flags_descriptor);
        state->active_tiles[i] = wgpuDeviceCreateBuffer(device, &tiles_descriptor);
        state->indirect_args[i] = wgpuDeviceCreateBuffer(device, &indirect_descriptor);
    }
    if (!state->uniforms || !state->cells[0] || !state->cells[1] ||
        !state->tile_flags[0] || !state->tile_flags[1] ||
        !state->active_tiles[0] || !state->active_tiles[1] ||
        !state->indirect_args[0] || !state->indirect_args[1]) goto fail;

    /* 5. CPU上の初期盤面と候補タイルをGPUへコピーする。 */
    WGPUQueue queue = wgpuDeviceGetQueue(device);
    if (!queue) goto fail;
    wgpuQueueWriteBuffer(queue, state->cells[0], 0, initial_words, GRID_BYTES);
    wgpuQueueWriteBuffer(queue, state->cells[1], 0, initial_words, GRID_BYTES);
    upload_initial_activity(queue, state, initial_tiles, initial_tile_count);
    wgpuQueueRelease(queue);

    /* 6. pipelineが要求する配線図Layoutを取り出し、BindGroupを作る。 */
    WGPUBindGroupLayout compute_layout =
        wgpuComputePipelineGetBindGroupLayout(state->step_one_pipeline, 0);
    WGPUBindGroupLayout compute_eight_layout =
        wgpuComputePipelineGetBindGroupLayout(state->step_eight_pipeline, 0);
    WGPUBindGroupLayout clear_layout =
        wgpuComputePipelineGetBindGroupLayout(state->clear_pipeline, 0);
    WGPUBindGroupLayout render_layout =
        wgpuRenderPipelineGetBindGroupLayout(state->render_pipeline, 0);
    if (!compute_layout || !compute_eight_layout || !clear_layout || !render_layout)
        goto fail_layout;

    /*
     * 自動生成Layoutは、見た目が同じbindingでもpipelineごとに別物の場合がある。
     * そのためstep_one用とstep_eight用を、それぞれ自身のLayoutから作る。
     * 添字0はcells[0]→cells[1]、添字1は逆向きの配線である。
     */
    state->step_one_groups[0] = make_step_group(
        device, compute_layout, state->uniforms,
        state->cells[0], state->cells[1], state->active_tiles[0], state->tile_flags[1],
        state->active_tiles[1], state->indirect_args[1]);
    state->step_one_groups[1] = make_step_group(
        device, compute_layout, state->uniforms,
        state->cells[1], state->cells[0], state->active_tiles[1], state->tile_flags[0],
        state->active_tiles[0], state->indirect_args[0]);
    state->step_eight_groups[0] = make_step_group(
        device, compute_eight_layout, state->uniforms,
        state->cells[0], state->cells[1], state->active_tiles[0], state->tile_flags[1],
        state->active_tiles[1], state->indirect_args[1]);
    state->step_eight_groups[1] = make_step_group(
        device, compute_eight_layout, state->uniforms,
        state->cells[1], state->cells[0], state->active_tiles[1], state->tile_flags[0],
        state->active_tiles[0], state->indirect_args[0]);
    state->clear_groups[0] = make_clear_group(
        device, clear_layout, state->uniforms, state->cells[0], state->active_tiles[0]);
    state->clear_groups[1] = make_clear_group(
        device, clear_layout, state->uniforms, state->cells[1], state->active_tiles[1]);
    state->render_groups[0] = make_render_group(device, render_layout,
                                                 state->uniforms, state->cells[0]);
    state->render_groups[1] = make_render_group(device, render_layout,
                                                 state->uniforms, state->cells[1]);
    wgpuBindGroupLayoutRelease(compute_layout);
    wgpuBindGroupLayoutRelease(compute_eight_layout);
    wgpuBindGroupLayoutRelease(clear_layout);
    wgpuBindGroupLayoutRelease(render_layout);
    if (!state->step_one_groups[0] || !state->step_one_groups[1] ||
        !state->step_eight_groups[0] || !state->step_eight_groups[1] ||
        !state->clear_groups[0] || !state->clear_groups[1] ||
        !state->render_groups[0] || !state->render_groups[1]) goto fail;

    /* すべて成功。最初はcells[0]を現在面としてLispへstateを返す。 */
    state->current = 0;
    state->present_mode = WGPUPresentMode_Fifo;
    state->immediate_supported = immediate_supported;
    *out_state = state;
    return device;

fail_layout:
    /* goto先を1か所に集めると、途中のどの段階で失敗しても解放漏れを防げる。 */
    if (compute_layout) wgpuBindGroupLayoutRelease(compute_layout);
    if (compute_eight_layout) wgpuBindGroupLayoutRelease(compute_eight_layout);
    if (clear_layout) wgpuBindGroupLayoutRelease(clear_layout);
    if (render_layout) wgpuBindGroupLayoutRelease(render_layout);
fail:
    WGPU_ReleaseLife(state);
    wgpuDeviceRelease(device);
    return NULL;
}

static int configure_surface(WGPUDevice device, WGPUSurface surface,
                             LifeState *state, uint32_t width, uint32_t height,
                             uint32_t request_immediate)
{
    /*
     * ウィンドウの大きさに合わせ、表示用画像の交換列（swapchain相当）を設定する。
     * Fifoは垂直同期に合わせて順番に表示し、画面の途中で絵が切れるのを防ぐ。
     * Immediateは垂直同期を待たない。未対応環境で要求された場合はFIFOへ戻し、
     * 戻り値1でLisp側へ知らせる。
     */
    if (!device || !surface || !state || !width || !height || request_immediate > 1u)
        return -1;
    bool fell_back = request_immediate && !state->immediate_supported;
    WGPUPresentMode present_mode = request_immediate && !fell_back
        ? WGPUPresentMode_Immediate : WGPUPresentMode_Fifo;
    WGPUSurfaceConfiguration config = {
        .device = device,
        .format = WGPUTextureFormat_BGRA8Unorm,
        .usage = WGPUTextureUsage_RenderAttachment,
        .presentMode = present_mode,
        .width = width,
        .height = height,
        .alphaMode = WGPUCompositeAlphaMode_Opaque
    };
    wgpuSurfaceConfigure(surface, &config);
    state->present_mode = present_mode;
    return fell_back ? 1 : 0;
}

LIFE_API int WGPU_UpdateSurface(WGPUDevice device, WGPUSurface surface,
                                uint32_t width, uint32_t height)
{
    /*
     * 旧版と同じ4引数ABIを保つ互換入口。古いLispからは従来どおりFIFOで設定する。
     * present modeを選ぶ新しいLispはWGPU_SetPresentModeを使う。
     */
    if (!device || !surface || !width || !height) return -1;
    WGPUSurfaceConfiguration config = {
        .device = device,
        .format = WGPUTextureFormat_BGRA8Unorm,
        .usage = WGPUTextureUsage_RenderAttachment,
        .presentMode = WGPUPresentMode_Fifo,
        .width = width,
        .height = height,
        .alphaMode = WGPUCompositeAlphaMode_Opaque
    };
    wgpuSurfaceConfigure(surface, &config);
    return 0;
}

LIFE_API int WGPU_SetPresentMode(WGPUDevice device, WGPUSurface surface,
                                 LifeState *state, uint32_t width, uint32_t height,
                                 uint32_t request_immediate)
{
    /* 新機能を別symbolにし、新旧DLLを混ぜても引数ずれでメモリを壊さないようにする。 */
    return configure_surface(device, surface, state, width, height,
                             request_immediate);
}

LIFE_API void WGPU_ResetLife(WGPUQueue queue, LifeState *state,
                             const uint32_t *initial_words,
                             uint64_t word_count,
                             const uint32_t *initial_tiles,
                             uint32_t initial_tile_count)
{
    /* Rキー用。GPU資源は作り直さず、2枚の内容と候補数だけを初期状態へ戻す。 */
    if (!queue || !state || !initial_words || word_count != WORD_COUNT ||
        initial_tile_count > TILE_COUNT || (initial_tile_count && !initial_tiles))
        return;
    upload_initial_activity(queue, state, initial_tiles, initial_tile_count);
    wgpuQueueWriteBuffer(queue, state->cells[0], 0, initial_words, GRID_BYTES);
    wgpuQueueWriteBuffer(queue, state->cells[1], 0, initial_words, GRID_BYTES);
    state->current = 0;
}

static bool encode_step(WGPUCommandEncoder encoder, LifeState *state,
                        WGPUComputePipeline pipeline, WGPUBindGroup *groups,
                        bool sparse)
{
    /* -------------------- 1回または8回の世代更新命令を記録する --------------------
     * CommandEncoderはGPUへ渡す「作業手順書」。ここではまだ実行せず、命令を並べる。
     */
    uint32_t current = state->current;
    uint32_t next = current ^ 1u;
    /* 次回候補をこれから作り直すので、重複防止フラグを全面0にする。 */
    wgpuCommandEncoderClearBuffer(encoder, state->tile_flags[next], 0,
                                  TILE_BUFFER_BYTES);

    if (sparse) {
        /* 出力先に以前残った候補タイルだけを消す。DENSEは全タイルを上書きするので不要。 */
        WGPUComputePassEncoder clear =
            wgpuCommandEncoderBeginComputePass(encoder, NULL);
        if (!clear) return false;
        wgpuComputePassEncoderSetPipeline(clear, state->clear_pipeline);
        wgpuComputePassEncoderSetBindGroup(clear, 0, state->clear_groups[next], 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(
            clear, state->indirect_args[next], 0);
        wgpuComputePassEncoderEnd(clear);
        wgpuComputePassEncoderRelease(clear);
    }

    /* indirectのy/z（どちらも1）は残し、新しく数えるxだけ0へ戻す。 */
    wgpuCommandEncoderClearBuffer(encoder, state->indirect_args[next], 0,
                                  sizeof(uint32_t));

    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    if (!pass) return false;
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, groups[current], 0, NULL);
    /* SPARSEは候補数だけ間接実行、DENSEは全TILE_COUNT個を直接実行する。 */
    if (sparse)
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(
            pass, state->indirect_args[current], 0);
    else
        wgpuComputePassEncoderDispatchWorkgroups(pass, TILE_COUNT, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    /* 次世代を書いた面を、新しい現在面にする。XOR 1で0と1が交互に切り替わる。 */
    state->current = next;
    return true;
}

LIFE_API int WGPU_AdvanceLife(WGPUDevice device, WGPUQueue queue,
                              LifeState *state, uint32_t steps,
                              uint32_t sparse_mode)
{
    /* -------------------- 画面を出さずに時計を早送りする --------------------
     * 起動時は、デジタル時計をシステムのローカル時刻まで進める。
     * DrawLifeを使うと途中の画面が毎回表示されてしまうので、
     * この関数はcomputeだけをGPUへ送り、renderとpresentは行わない。
     */
    if (!device || !queue || !state) return -1;
    if (steps == 0u) return 0;

    /* shaderは盤面の大きさとSPARSE/DENSEの別をuniformから読む。
     * カメラと画面サイズは描画しないので0でよい。
     */
    struct {
        float screen[4];
        float camera[4];
        uint32_t grid[4];
    } uniforms = {
        .grid = { GRID_WIDTH, GRID_HEIGHT, WORDS_PER_ROW,
                  sparse_mode ? 1u : 0u }
    };
    wgpuQueueWriteBuffer(queue, state->uniforms, 0, &uniforms, sizeof(uniforms));

    /* Lisp側は1,024世代までを1回で渡す。ここではその全部を
     * 1個のCommandEncoderへ記録する。
     */
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    if (!encoder) return -1;
    uint32_t original_current = state->current;

    while (steps > 0u) {
        uint32_t batch = steps >= 8u ? 8u : 1u;
        WGPUComputePipeline pipeline = batch == 8u ?
            state->step_eight_pipeline : state->step_one_pipeline;
        WGPUBindGroup *groups = batch == 8u ?
            state->step_eight_groups : state->step_one_groups;
        if (!encode_step(encoder, state, pipeline, groups, sparse_mode != 0u))
            goto fail;
        steps -= batch;
    }

    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, NULL);
    if (!command_buffer) goto fail;
    WGPUSubmissionIndex submission =
        wgpuQueueSubmitForIndex(queue, 1, &command_buffer);
    wgpuCommandBufferRelease(command_buffer);
    wgpuCommandEncoderRelease(encoder);
    return wgpuDevicePoll(device, true, &submission) ? 0 : -1;

fail:
    /* GPUへ送る前に失敗した場合は、入力面の番号も元に戻す。 */
    state->current = original_current;
    wgpuCommandEncoderRelease(encoder);
    return -1;
}

LIFE_API int WGPU_DrawLife(WGPUDevice device, WGPUQueue queue,
                           WGPUSurface surface, LifeState *state,
                           float width, float height, float center_x,
                           float center_y, float zoom, uint32_t steps,
                           uint32_t sparse_mode)
{
    /* -------------------- 世代更新と画面描画を1フレーム分まとめる -------------------- */
    if (!device || !queue || !surface || !state) return -1;
    /* Cの構造体配置をWGSLのParamsと同じ48バイトにし、毎フレームGPUへ送る。 */
    struct {
        float screen[4];
        float camera[4];
        uint32_t grid[4];
    } uniforms = {
        .screen = { width, height, zoom, 0.0f },
        .camera = { center_x, center_y, 0.0f, 0.0f },
        .grid = { GRID_WIDTH, GRID_HEIGHT, WORDS_PER_ROW, sparse_mode ? 1u : 0u }
    };
    wgpuQueueWriteBuffer(queue, state->uniforms, 0, &uniforms, sizeof(uniforms));

    /* Surfaceから「今回描いてよい画面用Texture」を1枚借りる。 */
    WGPUSurfaceTexture surface_texture = {0};
    wgpuSurfaceGetCurrentTexture(surface, &surface_texture);
    /* リサイズ等で古くなったSurfaceは再設定し、今回は描かずLispへ再試行を知らせる。 */
    if (surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Outdated ||
        surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Lost) {
        if (surface_texture.texture) wgpuTextureRelease(surface_texture.texture);
        configure_surface(device, surface, state,
                          (uint32_t)width, (uint32_t)height,
                          state->present_mode == WGPUPresentMode_Immediate ? 1u : 0u);
        return 1;
    }
    if (!surface_texture.texture)
        return surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Timeout ? 1 : -1;

    /* TextureViewはTextureの「描画先としての見方」。Encoderへ更新と描画を順に記録する。 */
    WGPUTexture texture = surface_texture.texture;
    WGPUTextureView view = wgpuTextureCreateView(texture, NULL);
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    if (!view || !encoder) goto fail_frame;
    uint32_t original_current = state->current;

    /* 8世代分は一括shaderで進め、端数だけ通常shaderへ渡す。 */
    while (steps >= 8u) {
        if (!encode_step(encoder, state, state->step_eight_pipeline,
                         state->step_eight_groups, sparse_mode != 0u))
            goto fail_encoder;
        steps -= 8u;
    }
    while (steps > 0u) {
        if (!encode_step(encoder, state, state->step_one_pipeline,
                         state->step_one_groups, sparse_mode != 0u))
            goto fail_encoder;
        steps--;
    }

    /* 計算後のcurrent面を、画面全体を覆う三角形1枚として描く。 */
    WGPURenderPassColorAttachment attachment = {0};
    attachment.view = view;
    attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    attachment.loadOp = WGPULoadOp_Clear;
    attachment.storeOp = WGPUStoreOp_Store;
    attachment.clearValue = (WGPUColor){ .r = 0.008, .g = 0.011, .b = 0.015, .a = 1.0 };
    WGPURenderPassDescriptor pass_descriptor = {
        .colorAttachmentCount = 1, .colorAttachments = &attachment
    };
    WGPURenderPassEncoder render =
        wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
    if (!render) goto fail_encoder;
    wgpuRenderPassEncoderSetPipeline(render, state->render_pipeline);
    wgpuRenderPassEncoderSetBindGroup(render, 0,
                                      state->render_groups[state->current], 0, NULL);
    wgpuRenderPassEncoderDraw(render, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(render);
    wgpuRenderPassEncoderRelease(render);

    /* 手順書を完成させ、Queueへ提出し、Surfaceを画面に表示する。 */
    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, NULL);
    if (!command_buffer) goto fail_encoder;
    wgpuQueueSubmit(queue, 1, &command_buffer);
    wgpuSurfacePresent(surface);
    wgpuCommandBufferRelease(command_buffer);
    wgpuCommandEncoderRelease(encoder);
    wgpuTextureViewRelease(view);
    wgpuTextureRelease(texture);
    return 0;

fail_encoder:
    /* 命令作成に失敗したら、CPU側のcurrent番号もフレーム開始時へ戻す。 */
    state->current = original_current;
    wgpuCommandEncoderRelease(encoder);
fail_frame:
    if (view) wgpuTextureViewRelease(view);
    wgpuTextureRelease(texture);
    return -1;
}

#ifdef LIFE_SHADER_VALIDATE
/*
 * -------------------- ウィンドウなしの検証プログラム --------------------
 * build時に-DLIFE_SHADER_VALIDATEを付けた場合だけ、ここから下をコンパイルする。
 * 製品用bridge.soにはmain関数を入れず、validate.sh用実行ファイルにはmainを入れるための仕組み。
 *
 * 検証では、同じ小さな盤面をCPUとGPUの両方で8世代進め、全ビットが同じか比較する。
 * 「pipelineを作れた」だけでなく「実際の計算結果が正しい」ことまで確かめる。
 */
typedef struct { LifeAtomicBool done; WGPUMapAsyncStatus status; } BufferMapRequest;

static void map_callback(WGPUMapAsyncStatus status, WGPUStringView message,
                         void *userdata1, void *userdata2)
{
    /* GPUバッファをCPUから読める状態へする非同期処理の完了通知。 */
    BufferMapRequest *request = userdata1;
    (void)message; (void)userdata2;
    request->status = status;
    life_atomic_store(&request->done, true);
}

static void cpu_life(const uint32_t *input, uint32_t *output,
                     uint32_t width, uint32_t height, uint32_t stride)
{
    /*
     * 比較用の分かりやすいCPU版Life。1セルずつ8近傍を数えるので遅いが、
     * GPU版とは違う素直な書き方にすることで、同じ間違いをしにくくする。
     */
    memset(output, 0, (size_t)stride * height * sizeof(*output));
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            unsigned neighbours = 0;
            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int sx = (int)x + dx, sy = (int)y + dy;
                if (sx >= 0 && sy >= 0 && sx < (int)width && sy < (int)height)
                    neighbours += (input[(uint32_t)sy * stride + (uint32_t)sx / 32u]
                                  >> ((uint32_t)sx & 31u)) & 1u;
            }
            bool alive = ((input[y * stride + x / 32u] >> (x & 31u)) & 1u) != 0;
            if (neighbours == 3u || (alive && neighbours == 2u))
                output[y * stride + x / 32u] |= 1u << (x & 31u);
        }
    }
}

typedef struct {
    WGPUBuffer flags[2];
    WGPUBuffer tiles[2];
    WGPUBuffer indirect[2];
    WGPUBindGroup step_groups[2];
    WGPUBindGroup clear_groups[2];
} TestSparseState;

static bool init_test_sparse(WGPUDevice device, WGPUQueue queue,
                             WGPUComputePipeline step_pipeline,
                             WGPUComputePipeline clear_pipeline,
                             WGPUBuffer uniform, WGPUBuffer cells[2],
                             uint64_t cell_bytes, uint32_t tile_count,
                             const uint32_t *initial_tiles,
                             uint32_t initial_tile_count,
                             TestSparseState *state)
{
    /* 小型テスト盤面専用に、候補タイル管理バッファとBindGroupを2組作る。 */
    uint64_t tile_bytes = (uint64_t)tile_count * sizeof(uint32_t);
    WGPUBufferDescriptor flags_desc = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, .size = tile_bytes
    };
    WGPUBufferDescriptor tiles_desc = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, .size = tile_bytes
    };
    WGPUBufferDescriptor indirect_desc = {
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect |
                 WGPUBufferUsage_CopyDst,
        .size = INDIRECT_BYTES
    };
    const uint32_t indirect[3] = { initial_tile_count, 1u, 1u };
    WGPUBindGroupLayout step_layout =
        wgpuComputePipelineGetBindGroupLayout(step_pipeline, 0);
    WGPUBindGroupLayout clear_layout =
        wgpuComputePipelineGetBindGroupLayout(clear_pipeline, 0);

    for (unsigned i = 0; i < 2; i++) {
        state->flags[i] = wgpuDeviceCreateBuffer(device, &flags_desc);
        state->tiles[i] = wgpuDeviceCreateBuffer(device, &tiles_desc);
        state->indirect[i] = wgpuDeviceCreateBuffer(device, &indirect_desc);
        if (initial_tile_count)
            wgpuQueueWriteBuffer(queue, state->tiles[i], 0, initial_tiles,
                                 (uint64_t)initial_tile_count * sizeof(*initial_tiles));
        wgpuQueueWriteBuffer(queue, state->indirect[i], 0, indirect, sizeof(indirect));
    }

    /* current=0/1の両方向について、製品コードと同じ入出力配線を再現する。 */
    for (unsigned current = 0; current < 2; current++) {
        unsigned next = current ^ 1u;
        WGPUBindGroupEntry step_entries[7] = {0};
        step_entries[0] = (WGPUBindGroupEntry){
            .binding = 0, .buffer = uniform, .size = UNIFORM_BYTES };
        step_entries[1] = (WGPUBindGroupEntry){
            .binding = 1, .buffer = cells[current], .size = cell_bytes };
        step_entries[2] = (WGPUBindGroupEntry){
            .binding = 2, .buffer = cells[next], .size = cell_bytes };
        step_entries[3] = (WGPUBindGroupEntry){
            .binding = 3, .buffer = state->tiles[current], .size = tile_bytes };
        step_entries[4] = (WGPUBindGroupEntry){
            .binding = 4, .buffer = state->flags[next], .size = tile_bytes };
        step_entries[5] = (WGPUBindGroupEntry){
            .binding = 5, .buffer = state->tiles[next], .size = tile_bytes };
        step_entries[6] = (WGPUBindGroupEntry){
            .binding = 6, .buffer = state->indirect[next], .size = INDIRECT_BYTES };
        WGPUBindGroupDescriptor step_desc = {
            .layout = step_layout, .entryCount = 7, .entries = step_entries
        };
        state->step_groups[current] = wgpuDeviceCreateBindGroup(device, &step_desc);

        WGPUBindGroupEntry clear_entries[3] = {0};
        clear_entries[0] = (WGPUBindGroupEntry){
            .binding = 0, .buffer = uniform, .size = UNIFORM_BYTES };
        clear_entries[1] = (WGPUBindGroupEntry){
            .binding = 2, .buffer = cells[current], .size = cell_bytes };
        clear_entries[2] = (WGPUBindGroupEntry){
            .binding = 3, .buffer = state->tiles[current], .size = tile_bytes };
        WGPUBindGroupDescriptor clear_desc = {
            .layout = clear_layout, .entryCount = 3, .entries = clear_entries
        };
        state->clear_groups[current] =
            wgpuDeviceCreateBindGroup(device, &clear_desc);
    }
    wgpuBindGroupLayoutRelease(clear_layout);
    wgpuBindGroupLayoutRelease(step_layout);
    return state->step_groups[0] && state->step_groups[1] &&
           state->clear_groups[0] && state->clear_groups[1];
}

static uint32_t build_test_active_tiles(const uint32_t *words,
                                        uint32_t height, uint32_t stride,
                                        uint32_t *tiles)
{
    /*
     * テスト入力からCPU上で候補タイルを作る独立実装。
     * 生セルを含むタイルを探し、その周囲3×3へflagsを立て、番号一覧へ変換する。
     */
    uint32_t columns = (stride + CORE_WORDS - 1u) / CORE_WORDS;
    uint32_t rows = (height + CORE_ROWS - 1u) / CORE_ROWS;
    uint32_t count = columns * rows;
    uint8_t *flags = calloc(count, sizeof(*flags));
    if (!flags) return UINT32_MAX;
    for (uint32_t tile_y = 0; tile_y < rows; tile_y++) {
        uint32_t y_end = (tile_y + 1u) * CORE_ROWS;
        if (y_end > height) y_end = height;
        for (uint32_t tile_x = 0; tile_x < columns; tile_x++) {
            uint32_t word_end = (tile_x + 1u) * CORE_WORDS;
            if (word_end > stride) word_end = stride;
            bool alive = false;
            for (uint32_t y = tile_y * CORE_ROWS; y < y_end && !alive; y++)
                for (uint32_t x = tile_x * CORE_WORDS; x < word_end; x++)
                    if (words[(uint64_t)y * stride + x]) { alive = true; break; }
            if (!alive) continue;
            for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                int x = (int)tile_x + dx, y = (int)tile_y + dy;
                if (x >= 0 && y >= 0 && x < (int)columns && y < (int)rows)
                    flags[(uint32_t)y * columns + (uint32_t)x] = 1u;
            }
        }
    }
    uint32_t active_count = 0;
    for (uint32_t i = 0; i < count; i++) if (flags[i]) tiles[active_count++] = i;
    free(flags);
    return active_count;
}

static void encode_test_step(WGPUCommandEncoder encoder,
                             WGPUComputePipeline step_pipeline,
                             WGPUComputePipeline clear_pipeline,
                             uint32_t tile_count, TestSparseState *state,
                             unsigned current, bool sparse)
{
    /* 製品のencode_stepと同じ順序を、小さい任意サイズの盤面向けに記録する。 */
    unsigned next = current ^ 1u;
    uint64_t tile_bytes = (uint64_t)tile_count * sizeof(uint32_t);
    wgpuCommandEncoderClearBuffer(encoder, state->flags[next], 0, tile_bytes);
    if (sparse) {
        WGPUComputePassEncoder clear =
            wgpuCommandEncoderBeginComputePass(encoder, NULL);
        wgpuComputePassEncoderSetPipeline(clear, clear_pipeline);
        wgpuComputePassEncoderSetBindGroup(clear, 0, state->clear_groups[next], 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(
            clear, state->indirect[next], 0);
        wgpuComputePassEncoderEnd(clear);
        wgpuComputePassEncoderRelease(clear);
    }
    wgpuCommandEncoderClearBuffer(encoder, state->indirect[next], 0, sizeof(uint32_t));
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    wgpuComputePassEncoderSetPipeline(pass, step_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, state->step_groups[current], 0, NULL);
    if (sparse)
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(
            pass, state->indirect[current], 0);
    else
        wgpuComputePassEncoderDispatchWorkgroups(pass, tile_count, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
}

static void release_test_sparse(TestSparseState *state)
{
    /* テストだけで作ったGPU資源を解放する。 */
    for (unsigned i = 0; i < 2; i++) {
        if (state->step_groups[i]) wgpuBindGroupRelease(state->step_groups[i]);
        if (state->clear_groups[i]) wgpuBindGroupRelease(state->clear_groups[i]);
        if (state->indirect[i]) wgpuBufferRelease(state->indirect[i]);
        if (state->tiles[i]) wgpuBufferRelease(state->tiles[i]);
        if (state->flags[i]) wgpuBufferRelease(state->flags[i]);
    }
}

int main(int argc, char **argv)
{
    /* validate.shからWGSLファイル名を1個受け取り、全文をC文字列として読む。 */
    if (argc != 2) return 2;
    FILE *file = fopen(argv[1], "rb");
    if (!file) return 2;
    if (fseek(file, 0, SEEK_END) != 0) return 2;
    long size = ftell(file);
    if (size < 0 || fseek(file, 0, SEEK_SET) != 0) return 2;
    char *code = malloc((size_t)size + 1u);
    if (!code || fread(code, 1, (size_t)size, file) != (size_t)size) return 2;
    code[size] = '\0';
    fclose(file);

    /* Surfaceは作らず、画面表示に対応していないGPUでも計算検証できるようにする。 */
    WGPUInstance instance = WGPU_CreateInstance(1);
    AdapterRequest adapter_request = {0};
    WGPURequestAdapterOptions options = { .powerPreference = WGPUPowerPreference_HighPerformance };
    WGPURequestAdapterCallbackInfo adapter_info = {
        .callback = adapter_callback, .userdata1 = &adapter_request,
        .mode = WGPUCallbackMode_AllowSpontaneous
    };
    wgpuInstanceRequestAdapter(instance, &options, adapter_info);
    while (!life_atomic_load(&adapter_request.done)) SDL_Delay(1);
    if (!adapter_request.adapter) return 3;
    DeviceRequest device_request = {0};
    WGPUDeviceDescriptor device_descriptor = {0};
    WGPURequestDeviceCallbackInfo device_info = {
        .callback = device_callback, .userdata1 = &device_request,
        .mode = WGPUCallbackMode_AllowSpontaneous
    };
    wgpuAdapterRequestDevice(adapter_request.adapter, &device_descriptor, device_info);
    while (!life_atomic_load(&device_request.done)) SDL_Delay(1);
    if (!device_request.device) return 3;

    /* まず4本のpipelineを作り、WGSLの文法とbindingが妥当か確認する。 */
    WGPUShaderModule module = make_shader(device_request.device, code);
    WGPUComputePipeline one = make_compute_pipeline(device_request.device, module, "step_one");
    WGPUComputePipeline eight = make_compute_pipeline(device_request.device, module, "step_eight");
    WGPUComputePipeline clear =
        make_compute_pipeline(device_request.device, module, "clear_tiles");
    WGPURenderPipeline render = make_render_pipeline(device_request.device, module);
    bool ok = module && one && eight && clear && render;
    if (ok) {
        /*
         * テスト盤面は横1056セル（33 words）×縦288セル。
         * 4 words/128行のタイル境界を両方向にまたぐサイズをわざと選んでいる。
         */
        enum { TEST_WIDTH = 1056, TEST_HEIGHT = 288, TEST_STRIDE = 33,
               TEST_GENERATIONS = 8, DENSE_GENERATIONS = 4,
               TEST_WORDS = TEST_STRIDE * TEST_HEIGHT,
               TEST_TILE_COLUMNS = (TEST_STRIDE + CORE_WORDS - 1) / CORE_WORDS,
               TEST_TILE_ROWS = (TEST_HEIGHT + CORE_ROWS - 1) / CORE_ROWS,
               TEST_TILE_COUNT = TEST_TILE_COLUMNS * TEST_TILE_ROWS };
        const uint64_t test_bytes = TEST_WORDS * sizeof(uint32_t);
        uint32_t initial[TEST_WORDS] = {0}, cpu_a[TEST_WORDS], cpu_b[TEST_WORDS];
        uint32_t seed = 0x12345678u;
        /* 疎な疑似乱数パターンをタイル境界へ置き、境界越しの誕生も検査する。 */
        for (unsigned y = 120; y < 137; y++) for (unsigned x = 2; x < 7; x++) {
            seed = seed * 1664525u + 1013904223u;
            initial[y * TEST_STRIDE + x] = seed & (seed >> 3u) & (seed >> 9u);
        }
        /* CPU参照結果を8世代分作る。aとbを交互に使う単純なダブルバッファ。 */
        memcpy(cpu_a, initial, sizeof(initial));
        for (unsigned i = 0; i < TEST_GENERATIONS; i++) {
            cpu_life(cpu_a, cpu_b, TEST_WIDTH, TEST_HEIGHT, TEST_STRIDE);
            memcpy(cpu_a, cpu_b, sizeof(cpu_a));
        }

        /* SPARSE通常8回、SPARSE一括8回、DENSE→SPARSE切替の3経路を別バッファで走らせる。 */
        WGPUBufferDescriptor uniform_desc = {
            .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, .size = UNIFORM_BYTES
        };
        WGPUBufferDescriptor storage_desc = {
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
            .size = test_bytes
        };
        WGPUBufferDescriptor staging_desc = {
            .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
            .size = test_bytes * 3u
        };
        WGPUBuffer sparse_uniform =
            wgpuDeviceCreateBuffer(device_request.device, &uniform_desc);
        WGPUBuffer dense_uniform =
            wgpuDeviceCreateBuffer(device_request.device, &uniform_desc);
        WGPUBuffer one_cells[2] = {
            wgpuDeviceCreateBuffer(device_request.device, &storage_desc),
            wgpuDeviceCreateBuffer(device_request.device, &storage_desc)
        };
        WGPUBuffer eight_cells[2] = {
            wgpuDeviceCreateBuffer(device_request.device, &storage_desc),
            wgpuDeviceCreateBuffer(device_request.device, &storage_desc)
        };
        WGPUBuffer dense_cells[2] = {
            wgpuDeviceCreateBuffer(device_request.device, &storage_desc),
            wgpuDeviceCreateBuffer(device_request.device, &storage_desc)
        };
        WGPUBuffer staging = wgpuDeviceCreateBuffer(device_request.device, &staging_desc);
        WGPUQueue queue = wgpuDeviceGetQueue(device_request.device);
        struct { float screen[4], camera[4]; uint32_t grid[4]; } params = {
            .screen = {640, 480, 1, 0}, .camera = {48, 32, 0, 0},
            .grid = {TEST_WIDTH, TEST_HEIGHT, TEST_STRIDE, 1}
        };
        wgpuQueueWriteBuffer(queue, sparse_uniform, 0, &params, sizeof(params));
        params.grid[3] = 0;
        wgpuQueueWriteBuffer(queue, dense_uniform, 0, &params, sizeof(params));
        for (unsigned i = 0; i < 2; i++) {
            wgpuQueueWriteBuffer(queue, one_cells[i], 0, initial, test_bytes);
            wgpuQueueWriteBuffer(queue, eight_cells[i], 0, initial, test_bytes);
            wgpuQueueWriteBuffer(queue, dense_cells[i], 0, initial, test_bytes);
        }
        TestSparseState one_state = {0}, eight_state = {0}, dense_state = {0};
        uint32_t initial_tiles[TEST_TILE_COUNT];
        uint32_t initial_tile_count = build_test_active_tiles(
            initial, TEST_HEIGHT, TEST_STRIDE, initial_tiles);
        ok = initial_tile_count != UINT32_MAX;
        ok = ok &&
             init_test_sparse(device_request.device, queue, one, clear,
                              sparse_uniform, one_cells, test_bytes, TEST_TILE_COUNT,
                              initial_tiles, initial_tile_count, &one_state) &&
             init_test_sparse(device_request.device, queue, eight, clear,
                              sparse_uniform, eight_cells, test_bytes, TEST_TILE_COUNT,
                              initial_tiles, initial_tile_count, &eight_state) &&
             init_test_sparse(device_request.device, queue, one, clear,
                              dense_uniform, dense_cells, test_bytes, TEST_TILE_COUNT,
                              initial_tiles, initial_tile_count, &dense_state);

        /* 切替テストは最初の4世代をDENSEで進める。 */
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device_request.device, NULL);
        for (unsigned i = 0; i < DENSE_GENERATIONS; i++)
            encode_test_step(encoder, one, clear, TEST_TILE_COUNT,
                             &dense_state, i & 1u, false);
        WGPUCommandBuffer dense_commands = wgpuCommandEncoderFinish(encoder, NULL);
        wgpuQueueSubmit(queue, 1, &dense_commands);
        wgpuCommandBufferRelease(dense_commands);
        wgpuCommandEncoderRelease(encoder);

        /* 同じセル/候補状態をリセットせず、残り4世代だけSPARSEへ切り替える。 */
        params.grid[3] = 1;
        wgpuQueueWriteBuffer(queue, dense_uniform, 0, &params, sizeof(params));
        encoder = wgpuDeviceCreateCommandEncoder(device_request.device, NULL);
        for (unsigned i = 0; i < TEST_GENERATIONS; i++) {
            encode_test_step(encoder, one, clear, TEST_TILE_COUNT,
                             &one_state, i & 1u, true);
        }
        for (unsigned i = DENSE_GENERATIONS; i < TEST_GENERATIONS; i++) {
            encode_test_step(encoder, one, clear, TEST_TILE_COUNT,
                             &dense_state, i & 1u, true);
        }
        for (unsigned i = 0; i < TEST_GENERATIONS / 8; i++) {
            encode_test_step(encoder, eight, clear, TEST_TILE_COUNT,
                             &eight_state, i & 1u, true);
        }
        /* 3経路のGPU結果を、CPUから読めるstagingバッファへ連続してコピーする。 */
        wgpuCommandEncoderCopyBufferToBuffer(encoder, one_cells[0], 0, staging, 0, test_bytes);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, eight_cells[1], 0, staging, test_bytes, test_bytes);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, dense_cells[0], 0, staging,
                                             test_bytes * 2u, test_bytes);
        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
        wgpuQueueSubmit(queue, 1, &commands);

        /* map完了を待ち、3つともCPU参照結果と1バイトずつ完全一致するか調べる。 */
        BufferMapRequest map = {0};
        WGPUBufferMapCallbackInfo map_info = {
            .mode = WGPUCallbackMode_AllowSpontaneous,
            .callback = map_callback, .userdata1 = &map
        };
        wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, test_bytes * 3u, map_info);
        while (!life_atomic_load(&map.done))
            wgpuDevicePoll(device_request.device, true, NULL);
        const uint32_t *gpu = wgpuBufferGetConstMappedRange(staging, 0, test_bytes * 3u);
        ok = map.status == WGPUMapAsyncStatus_Success && gpu &&
             memcmp(gpu, cpu_a, test_bytes) == 0 &&
             memcmp((const uint8_t *)gpu + test_bytes, cpu_a, test_bytes) == 0 &&
             memcmp((const uint8_t *)gpu + test_bytes * 2u, cpu_a, test_bytes) == 0;
        if (gpu) wgpuBufferUnmap(staging);

        wgpuCommandBufferRelease(commands); wgpuCommandEncoderRelease(encoder);
        release_test_sparse(&dense_state);
        release_test_sparse(&eight_state);
        release_test_sparse(&one_state);
        wgpuQueueRelease(queue); wgpuBufferRelease(staging);
        wgpuBufferRelease(dense_uniform); wgpuBufferRelease(sparse_uniform);
        for (unsigned i = 0; i < 2; i++) {
            wgpuBufferRelease(dense_cells[i]);
            wgpuBufferRelease(eight_cells[i]);
            wgpuBufferRelease(one_cells[i]);
        }
    }
    /* 成否に関係なく、作成した順と逆向きにすべて解放する。 */
    if (render) wgpuRenderPipelineRelease(render);
    if (clear) wgpuComputePipelineRelease(clear);
    if (eight) wgpuComputePipelineRelease(eight);
    if (one) wgpuComputePipelineRelease(one);
    if (module) wgpuShaderModuleRelease(module);
    wgpuDeviceRelease(device_request.device);
    wgpuAdapterRelease(adapter_request.adapter);
    wgpuInstanceRelease(instance);
    free(code);
    fprintf(stderr, "WGSL validation: %s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
#endif
