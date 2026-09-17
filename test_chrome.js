// Browser control layer for the existing shader.wgsl.
// The WGSL is fetched as-is: this file reproduces bridge.c and lifegame.lisp in WebGPU/JavaScript.

const GRID_WIDTH = 20_000;
const GRID_HEIGHT = 20_000;
const WORDS_PER_ROW = Math.ceil(GRID_WIDTH / 32);
const WORD_COUNT = WORDS_PER_ROW * GRID_HEIGHT;
const GRID_BYTES = WORD_COUNT * 4;

const CORE_WORDS = 4;
const CORE_ROWS = 128;
const TILE_COLUMNS = Math.ceil(WORDS_PER_ROW / CORE_WORDS);
const TILE_ROWS = Math.ceil(GRID_HEIGHT / CORE_ROWS);
const TILE_COUNT = TILE_COLUMNS * TILE_ROWS;
const TILE_BUFFER_BYTES = TILE_COUNT * 4;
const INDIRECT_BYTES = 3 * 4;
const UNIFORM_BYTES = 48;

const MAX_STEPS_PER_FRAME = 256;
const SYNC_STEPS_PER_BATCH = 1024;
const MIN_ZOOM_LEVEL = -14;
const MAX_ZOOM_LEVEL = 19;
const INITIAL_ZOOM_LEVEL = 16;

const CLOCK_GENERATIONS_PER_SECOND = 192;
const CLOCK_SNAPSHOT_INTERVAL = 115_200;
const CLOCK_SNAPSHOT_BASE_GENERATION = 24_883_200;
const CLOCK_GENERATIONS_PER_DAY = 24 * 60 * 60 * CLOCK_GENERATIONS_PER_SECOND;
const CLOCK_DISPLAY_LAG_GENERATIONS = 11_520;
const CLOCK_CORRECTION_MS = 300_000;

const SPEED_LEVELS = Object.freeze([
  1, 2, 3, 4, 5,
  10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
  110, 120, 130, 140, 150, 160, 170, 180, 190, 192, 200,
  210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
]);

const canvas = document.querySelector("#life");
const loading = document.querySelector("#loading");
const loadingTitle = document.querySelector("#loading-title");
const loadingDetail = document.querySelector("#loading-detail");
const hud = document.querySelector("#hud");
const statsElement = document.querySelector("#stats");
const messageElement = document.querySelector("#message");
const controls = document.querySelector("#controls");
const pauseButton = document.querySelector("#pause");

const state = {
  adapter: null,
  device: null,
  context: null,
  format: null,
  queue: null,
  pipelines: null,
  buffers: null,
  groups: null,
  current: 0,

  initialWords: null,
  initialTiles: null,
  initialGeneration: 0,
  sourceName: "",

  width: 1,
  height: 1,
  cssWidth: 1,
  cssHeight: 1,
  pixelRatio: 1,
  centerX: GRID_WIDTH / 2,
  centerY: GRID_HEIGHT / 2,
  zoomLevel: INITIAL_ZOOM_LEVEL,
  zoom: zoomForLevel(INITIAL_ZOOM_LEVEL),
  mouseX: 0,
  mouseY: 0,
  dragging: false,

  sparseMode: true,
  observedSparseMode: true,
  paused: false,
  singleStep: 0,
  speed: CLOCK_GENERATIONS_PER_SECOND,
  generation: 0,
  accumulator: 0,
  syncing: true,
  syncOriginWallMs: 0,
  syncOriginGeneration: 0,
  nextClockCorrectionMs: null,

  running: true,
  ready: false,
  lastFrameTime: performance.now(),
  statsStart: performance.now(),
  statsGeneration: 0,
  statsFrameCount: 0,
  measuredSpeed: 0,
  measuredFps: 0,
  lastHudTime: 0,
  message: "",
};

function zoomForLevel(level) {
  return Math.min(64, Math.max(0.05, 1.25 ** level));
}

// UIへ表示するzoomは「cells / CSS pixel」。WGSLはCanvas内部の物理ピクセルを
// 受け取るため、描画直前だけdevicePixelRatio相当の倍率で割る。
function shaderZoom(zoom = state.zoom) {
  return zoom / state.pixelRatio;
}

function setLoading(title, detail = "") {
  loadingTitle.textContent = title;
  loadingDetail.textContent = detail;
}

function setMessage(message) {
  state.message = message;
  messageElement.textContent = message;
}

function formatInteger(value) {
  return Math.trunc(value).toLocaleString("en-US");
}

function resetStats(now = performance.now()) {
  state.statsStart = now;
  state.statsGeneration = state.generation;
  state.statsFrameCount = 0;
  state.measuredSpeed = 0;
  state.measuredFps = 0;
}

function updateHud(now = performance.now()) {
  if (now - state.lastHudTime < 250) return;
  state.lastHudTime = now;
  const target = state.speed === CLOCK_GENERATIONS_PER_SECOND ? "*192*" : state.speed.toFixed(1);
  const mode = state.sparseMode ? "SPARSE" : "DENSE";
  const runState = state.paused ? "PAUSE" : "RUN";
  const title = `Life 20000² | ${mode} | gen ${formatInteger(state.generation)} | ${runState} | target ${target} gen/s | actual ${state.measuredSpeed.toFixed(1)} gen/s | ${state.measuredFps.toFixed(1)} fps | zoom ${state.zoom.toFixed(1)} cell/px`;
  document.title = title;
  statsElement.textContent = title;
  messageElement.textContent = state.message;
  pauseButton.textContent = state.paused ? "Run" : "Pause";
}

function positiveModulo(value, modulus) {
  return ((value % modulus) + modulus) % modulus;
}

function clockGenerationDifference(target, current) {
  const difference = positiveModulo(target - current, CLOCK_GENERATIONS_PER_DAY);
  return difference > CLOCK_GENERATIONS_PER_DAY / 2
    ? difference - CLOCK_GENERATIONS_PER_DAY
    : difference;
}

function localGenerationAt(date) {
  const seconds = date.getHours() * 3600 + date.getMinutes() * 60 + date.getSeconds();
  return CLOCK_SNAPSHOT_BASE_GENERATION
    + seconds * CLOCK_GENERATIONS_PER_SECOND
    + CLOCK_DISPLAY_LAG_GENERATIONS;
}

function snapshotInfo(date) {
  const hour = date.getHours();
  const snapshotMinute = Math.floor(date.getMinutes() / 10) * 10;
  const snapshotIndex = hour * 6 + snapshotMinute / 10;
  const generation = CLOCK_SNAPSHOT_BASE_GENERATION
    + snapshotIndex * CLOCK_SNAPSHOT_INTERVAL;
  // 現在のepoch時刻から端数だけを引く。夏時間切替付近でも「直前10分」を保つ。
  const elapsedMs = (
    (date.getMinutes() % 10) * 60 + date.getSeconds()
  ) * 1000 + date.getMilliseconds();
  const hh = String(hour).padStart(2, "0");
  const mm = String(snapshotMinute).padStart(2, "0");
  return {
    url: `clock-snapshot/clock-${hh}-${mm}.rle`,
    generation,
    originWallMs: date.getTime() - elapsedMs,
  };
}

async function fetchText(url, description) {
  const response = await fetch(url, { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`${description}の取得に失敗しました (${response.status} ${response.statusText}): ${url}`);
  }
  return response.text();
}

function parseRle(text) {
  const lines = text.split(/\r?\n/);
  let width = null;
  let height = null;
  const bodyLines = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === "" || trimmed.startsWith("#")) continue;
    if (width === null) {
      const match = trimmed.match(/x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)/i);
      if (match) {
        width = Number(match[1]);
        height = Number(match[2]);
        continue;
      }
    }
    bodyLines.push(trimmed);
  }

  if (width === null || height === null) throw new Error("RLE headerがありません。");
  if (width > GRID_WIDTH || height > GRID_HEIGHT) {
    throw new Error(`RLE ${width}x${height} は ${GRID_WIDTH}x${GRID_HEIGHT} の盤面に収まりません。`);
  }

  const words = new Uint32Array(WORD_COUNT);
  const occupied = new Uint8Array(TILE_COUNT);
  const offsetX = Math.floor((GRID_WIDTH - width) / 2);
  const offsetY = Math.floor((GRID_HEIGHT - height) / 2);
  const body = bodyLines.join("");
  let x = 0;
  let y = 0;
  let run = 0;
  let liveCount = 0;

  const takeRun = () => {
    const count = run === 0 ? 1 : run;
    run = 0;
    return count;
  };

  for (let index = 0; index < body.length; index += 1) {
    const character = body[index];
    const code = body.charCodeAt(index);
    if (code >= 48 && code <= 57) {
      run = run * 10 + code - 48;
      continue;
    }
    if (character === "b" || character === "B" || character === "o" || character === "O") {
      const count = takeRun();
      if (x + count > width || y >= height) {
        throw new Error(`RLE bodyが宣言寸法を超えました: (${x}, ${y})`);
      }
      if (character === "o" || character === "O") {
        for (let cell = 0; cell < count; cell += 1) {
          const gridX = offsetX + x + cell;
          const gridY = offsetY + y;
          const wordX = gridX >>> 5;
          words[gridY * WORDS_PER_ROW + wordX] |= (1 << (gridX & 31)) >>> 0;
          const tileX = Math.floor(wordX / CORE_WORDS);
          const tileY = Math.floor(gridY / CORE_ROWS);
          occupied[tileY * TILE_COLUMNS + tileX] = 1;
        }
        liveCount += count;
      }
      x += count;
      continue;
    }
    if (character === "$") {
      y += takeRun();
      x = 0;
      if (y > height) throw new Error("RLE bodyが宣言された高さを超えました。");
      continue;
    }
    if (character === "!") break;
    if (!/\s/.test(character)) throw new Error(`不正なRLE文字: ${JSON.stringify(character)}`);
  }

  const candidates = new Uint8Array(TILE_COUNT);
  for (let tile = 0; tile < TILE_COUNT; tile += 1) {
    if (occupied[tile] === 0) continue;
    const tileX = tile % TILE_COLUMNS;
    const tileY = Math.floor(tile / TILE_COLUMNS);
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        const nextX = tileX + dx;
        const nextY = tileY + dy;
        if (nextX >= 0 && nextX < TILE_COLUMNS && nextY >= 0 && nextY < TILE_ROWS) {
          candidates[nextY * TILE_COLUMNS + nextX] = 1;
        }
      }
    }
  }

  let tileCount = 0;
  for (const candidate of candidates) tileCount += candidate;
  const tiles = new Uint32Array(tileCount);
  let tileOutput = 0;
  for (let tile = 0; tile < TILE_COUNT; tile += 1) {
    if (candidates[tile]) tiles[tileOutput++] = tile;
  }

  return { words, tiles, liveCount, width, height };
}

function validateSnapshotGeneration(text, expectedGeneration, sourceName) {
  const match = text.match(/^#C Generation=(\d+)\b.*$/m);
  if (!match) throw new Error(`スナップショットにGenerationコメントがありません: ${sourceName}`);
  const actual = Number(match[1]);
  if (actual !== expectedGeneration) {
    throw new Error(`スナップショット世代が違います: expected ${expectedGeneration}, got ${actual}`);
  }
}

function createInitializedBuffer(device, label, usage, source) {
  const buffer = device.createBuffer({
    label,
    size: source.byteLength,
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(source.buffer, source.byteOffset, source.byteLength),
  );
  buffer.unmap();
  return buffer;
}

function makeEntry(binding, buffer, size) {
  return { binding, resource: { buffer, offset: 0, size } };
}

async function createGpu(shaderCode, initialWords, initialTiles) {
  if (!navigator.gpu) {
    throw new Error("WebGPUを利用できません。最新版のChromeを使い、localhostまたはHTTPSから開いてください。");
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("WebGPU hardware adapterを取得できませんでした。");
  if (adapter.limits.maxStorageBufferBindingSize < GRID_BYTES) {
    throw new Error(`GPUのmaxStorageBufferBindingSizeが不足しています (${formatInteger(adapter.limits.maxStorageBufferBindingSize)} < ${formatInteger(GRID_BYTES)})。`);
  }
  if (adapter.limits.maxBufferSize < GRID_BYTES) {
    throw new Error(`GPUのmaxBufferSizeが不足しています (${formatInteger(adapter.limits.maxBufferSize)} < ${formatInteger(GRID_BYTES)})。`);
  }

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: GRID_BYTES,
      maxBufferSize: GRID_BYTES,
    },
  });
  const queue = device.queue;
  device.lost.then((info) => {
    state.running = false;
    showFatalError(new Error(`WebGPU device lost: ${info.message || info.reason}`));
  });
  device.addEventListener("uncapturederror", (event) => {
    console.error("WebGPU uncaptured error", event.error);
    setMessage(`WebGPU error: ${event.error.message}`);
  });

  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("canvasのWebGPU contextを取得できませんでした。");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const shader = device.createShaderModule({ label: "unchanged shader.wgsl", code: shaderCode });
  const compilation = await shader.getCompilationInfo();
  const shaderErrors = compilation.messages.filter((message) => message.type === "error");
  if (shaderErrors.length > 0) {
    const detail = shaderErrors
      .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`)
      .join("\n");
    throw new Error(`shader.wgslのコンパイルに失敗しました。\n${detail}`);
  }

  const [stepOne, stepEight, clearTiles, render] = await Promise.all([
    device.createComputePipelineAsync({
      label: "step_one",
      layout: "auto",
      compute: { module: shader, entryPoint: "step_one" },
    }),
    device.createComputePipelineAsync({
      label: "step_eight",
      layout: "auto",
      compute: { module: shader, entryPoint: "step_eight" },
    }),
    device.createComputePipelineAsync({
      label: "clear_tiles",
      layout: "auto",
      compute: { module: shader, entryPoint: "clear_tiles" },
    }),
    device.createRenderPipelineAsync({
      label: "Life render",
      layout: "auto",
      vertex: { module: shader, entryPoint: "vs_main" },
      fragment: { module: shader, entryPoint: "fs_main", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    }),
  ]);

  const uniform = device.createBuffer({
    label: "uniforms",
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const cellUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const cells = [
    createInitializedBuffer(device, "cells[0]", cellUsage, initialWords),
    createInitializedBuffer(device, "cells[1]", cellUsage, initialWords),
  ];
  const tileFlags = [];
  const activeTiles = [];
  const indirectArgs = [];
  for (let index = 0; index < 2; index += 1) {
    tileFlags.push(device.createBuffer({
      label: `tile_flags[${index}]`,
      size: TILE_BUFFER_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }));
    activeTiles.push(device.createBuffer({
      label: `active_tiles[${index}]`,
      size: TILE_BUFFER_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }));
    indirectArgs.push(device.createBuffer({
      label: `indirect_args[${index}]`,
      size: INDIRECT_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    }));
  }

  const pipelines = { stepOne, stepEight, clearTiles, render };
  const buffers = { uniform, cells, tileFlags, activeTiles, indirectArgs };

  const makeStepGroups = (pipeline) => {
    const layout = pipeline.getBindGroupLayout(0);
    return [0, 1].map((current) => {
      const next = current ^ 1;
      return device.createBindGroup({
        layout,
        entries: [
          makeEntry(0, uniform, UNIFORM_BYTES),
          makeEntry(1, cells[current], GRID_BYTES),
          makeEntry(2, cells[next], GRID_BYTES),
          makeEntry(3, activeTiles[current], TILE_BUFFER_BYTES),
          makeEntry(4, tileFlags[next], TILE_BUFFER_BYTES),
          makeEntry(5, activeTiles[next], TILE_BUFFER_BYTES),
          makeEntry(6, indirectArgs[next], INDIRECT_BYTES),
        ],
      });
    });
  };

  const clearLayout = clearTiles.getBindGroupLayout(0);
  const clearGroups = [0, 1].map((index) => device.createBindGroup({
    layout: clearLayout,
    entries: [
      makeEntry(0, uniform, UNIFORM_BYTES),
      makeEntry(2, cells[index], GRID_BYTES),
      makeEntry(3, activeTiles[index], TILE_BUFFER_BYTES),
    ],
  }));

  const renderLayout = render.getBindGroupLayout(0);
  const renderGroups = [0, 1].map((index) => device.createBindGroup({
    layout: renderLayout,
    entries: [
      makeEntry(0, uniform, UNIFORM_BYTES),
      makeEntry(1, cells[index], GRID_BYTES),
    ],
  }));

  state.adapter = adapter;
  state.device = device;
  state.context = context;
  state.format = format;
  state.queue = queue;
  state.pipelines = pipelines;
  state.buffers = buffers;
  state.groups = {
    stepOne: makeStepGroups(stepOne),
    stepEight: makeStepGroups(stepEight),
    clear: clearGroups,
    render: renderGroups,
  };
  state.current = 0;
  uploadInitialActivity(initialTiles);
}

function uploadInitialActivity(tiles) {
  const indirect = new Uint32Array([tiles.length, 1, 1]);
  for (let index = 0; index < 2; index += 1) {
    if (tiles.length > 0) state.queue.writeBuffer(state.buffers.activeTiles[index], 0, tiles);
    state.queue.writeBuffer(state.buffers.indirectArgs[index], 0, indirect);
  }
}

function writeUniforms(width, height, centerX, centerY, zoom, sparse) {
  const data = new ArrayBuffer(UNIFORM_BYTES);
  const floats = new Float32Array(data);
  const integers = new Uint32Array(data);
  floats[0] = width;
  floats[1] = height;
  floats[2] = zoom;
  floats[3] = 0;
  floats[4] = centerX;
  floats[5] = centerY;
  floats[6] = 0;
  floats[7] = 0;
  integers[8] = GRID_WIDTH;
  integers[9] = GRID_HEIGHT;
  integers[10] = WORDS_PER_ROW;
  integers[11] = sparse ? 1 : 0;
  state.queue.writeBuffer(state.buffers.uniform, 0, data);
}

function encodeStep(encoder, pipeline, groups, sparse) {
  const current = state.current;
  const next = current ^ 1;
  encoder.clearBuffer(state.buffers.tileFlags[next], 0, TILE_BUFFER_BYTES);

  if (sparse) {
    const clearPass = encoder.beginComputePass({ label: "clear previous output tiles" });
    clearPass.setPipeline(state.pipelines.clearTiles);
    clearPass.setBindGroup(0, state.groups.clear[next]);
    clearPass.dispatchWorkgroupsIndirect(state.buffers.indirectArgs[next], 0);
    clearPass.end();
  }

  // y and z remain 1. Only the workgroup count at byte 0 is regenerated.
  encoder.clearBuffer(state.buffers.indirectArgs[next], 0, 4);
  const pass = encoder.beginComputePass({ label: pipeline.label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, groups[current]);
  if (sparse) {
    pass.dispatchWorkgroupsIndirect(state.buffers.indirectArgs[current], 0);
  } else {
    pass.dispatchWorkgroups(TILE_COUNT, 1, 1);
  }
  pass.end();
  state.current = next;
}

async function advanceLife(steps, waitForCompletion = false) {
  if (steps <= 0) return;
  writeUniforms(0, 0, 0, 0, 0, state.sparseMode);
  const encoder = state.device.createCommandEncoder({ label: `advance ${steps} generations` });
  let remaining = steps;
  while (remaining > 0) {
    const batch = remaining >= 8 ? 8 : 1;
    if (batch === 8) {
      encodeStep(encoder, state.pipelines.stepEight, state.groups.stepEight, state.sparseMode);
    } else {
      encodeStep(encoder, state.pipelines.stepOne, state.groups.stepOne, state.sparseMode);
    }
    remaining -= batch;
  }
  state.queue.submit([encoder.finish()]);
  if (waitForCompletion) await state.queue.onSubmittedWorkDone();
}

function drawLife(steps) {
  writeUniforms(
    state.width,
    state.height,
    state.centerX,
    state.centerY,
    shaderZoom(),
    state.sparseMode,
  );
  const encoder = state.device.createCommandEncoder({ label: "compute and draw frame" });
  let remaining = steps;
  while (remaining >= 8) {
    encodeStep(encoder, state.pipelines.stepEight, state.groups.stepEight, state.sparseMode);
    remaining -= 8;
  }
  while (remaining > 0) {
    encodeStep(encoder, state.pipelines.stepOne, state.groups.stepOne, state.sparseMode);
    remaining -= 1;
  }

  const view = state.context.getCurrentTexture().createView();
  const renderPass = encoder.beginRenderPass({
    label: "Life render",
    colorAttachments: [{
      view,
      clearValue: { r: 0.008, g: 0.011, b: 0.015, a: 1 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  renderPass.setPipeline(state.pipelines.render);
  renderPass.setBindGroup(0, state.groups.render[state.current]);
  renderPass.draw(3);
  renderPass.end();
  state.queue.submit([encoder.finish()]);
}

async function resetGrid() {
  if (!state.ready) return;
  uploadInitialActivity(state.initialTiles);
  state.queue.writeBuffer(state.buffers.cells[0], 0, state.initialWords);
  state.queue.writeBuffer(state.buffers.cells[1], 0, state.initialWords);
  state.current = 0;
  state.generation = state.initialGeneration;
  state.accumulator = 0;
  state.singleStep = 0;
  state.paused = true;
  resetStats();
  setMessage(`${state.sourceName} の初期状態へ戻しました。`);
}

function fullView() {
  const requiredZoom = 1.04 * Math.max(
    GRID_WIDTH / state.cssWidth,
    GRID_HEIGHT / state.cssHeight,
  );
  let level = MAX_ZOOM_LEVEL;
  for (let candidate = MIN_ZOOM_LEVEL; candidate <= MAX_ZOOM_LEVEL; candidate += 1) {
    if (zoomForLevel(candidate) >= requiredZoom) {
      level = candidate;
      break;
    }
  }
  state.centerX = GRID_WIDTH / 2;
  state.centerY = GRID_HEIGHT / 2;
  state.zoomLevel = level;
  state.zoom = zoomForLevel(level);
}

function changeSpeed(direction) {
  if (direction > 0) {
    state.speed = SPEED_LEVELS.find((value) => value > state.speed)
      ?? SPEED_LEVELS.at(-1);
  } else {
    state.speed = SPEED_LEVELS.findLast((value) => value < state.speed)
      ?? SPEED_LEVELS[0];
  }
  setMessage(`target ${state.speed} gen/s`);
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const scale = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(rect.width * scale));
  const height = Math.max(1, Math.round(rect.height * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  state.cssWidth = Math.max(1, rect.width);
  state.cssHeight = Math.max(1, rect.height);
  state.pixelRatio = scale;
  state.width = width;
  state.height = height;
}

function installInputHandlers() {
  new ResizeObserver(resizeCanvas).observe(canvas);
  // ブラウザ倍率の変更や、DPIの異なるモニターへの移動にも追従する。
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    state.dragging = true;
    canvas.classList.add("dragging");
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointerup", (event) => {
    state.dragging = false;
    canvas.classList.remove("dragging");
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointercancel", () => {
    state.dragging = false;
    canvas.classList.remove("dragging");
  });
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = state.width / rect.width;
    const scaleY = state.height / rect.height;
    state.mouseX = (event.clientX - rect.left) * scaleX;
    state.mouseY = (event.clientY - rect.top) * scaleY;
    if (state.dragging) {
      const renderZoom = shaderZoom();
      state.centerX -= event.movementX * scaleX * renderZoom;
      state.centerY -= event.movementY * scaleY * renderZoom;
    }
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    if (event.deltaY === 0) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = state.width / rect.width;
    const scaleY = state.height / rect.height;
    const mouseX = (event.clientX - rect.left) * scaleX;
    const mouseY = (event.clientY - rect.top) * scaleY;
    const currentShaderZoom = shaderZoom();
    const worldX = state.centerX + (mouseX - state.width / 2) * currentShaderZoom;
    const worldY = state.centerY + (mouseY - state.height / 2) * currentShaderZoom;
    const direction = event.deltaY < 0 ? -1 : 1;
    const newLevel = Math.min(MAX_ZOOM_LEVEL, Math.max(MIN_ZOOM_LEVEL, state.zoomLevel + direction));
    const newZoom = zoomForLevel(newLevel);
    const newShaderZoom = shaderZoom(newZoom);
    state.centerX = worldX - (mouseX - state.width / 2) * newShaderZoom;
    state.centerY = worldY - (mouseY - state.height / 2) * newShaderZoom;
    state.zoomLevel = newLevel;
    state.zoom = newZoom;
  }, { passive: false });

  window.addEventListener("keydown", (event) => {
    if (event.repeat || !state.ready) return;
    if (["Space", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.code)) event.preventDefault();
    switch (event.code) {
      case "Escape":
        state.running = false;
        state.paused = true;
        setMessage("停止しました。再開するにはページを再読み込みしてください。");
        updateHud();
        break;
      case "Space":
        state.paused = !state.paused;
        break;
      case "ArrowRight":
        state.paused = true;
        state.singleStep += 1;
        break;
      case "ArrowUp": changeSpeed(1); break;
      case "ArrowDown": changeSpeed(-1); break;
      case "KeyF": fullView(); break;
      case "KeyR": void resetGrid(); break;
      default: break;
    }
  });

  pauseButton.addEventListener("click", () => { state.paused = !state.paused; });
  document.querySelector("#step").addEventListener("click", () => {
    state.paused = true;
    state.singleStep += 1;
  });
  document.querySelector("#full").addEventListener("click", fullView);
  document.querySelector("#reset").addEventListener("click", () => void resetGrid());
}

async function synchronizeClock() {
  let lastReport = 0;
  for (;;) {
    const wallSeconds = Math.floor((Date.now() - state.syncOriginWallMs) / 1000);
    const target = state.syncOriginGeneration
      + wallSeconds * CLOCK_GENERATIONS_PER_SECOND
      + CLOCK_DISPLAY_LAG_GENERATIONS;
    const remaining = target - state.generation;
    if (remaining <= 0) break;
    const batch = Math.min(SYNC_STEPS_PER_BATCH, remaining);
    await advanceLife(batch, true);
    state.generation += batch;

    const now = performance.now();
    if (now - lastReport >= 100 || remaining <= batch) {
      lastReport = now;
      const complete = state.generation - state.syncOriginGeneration;
      const total = Math.max(1, target - state.syncOriginGeneration);
      setLoading(
        "ローカル時刻へ同期しています",
        `${formatInteger(complete)} / ${formatInteger(total)} generations (${(100 * complete / total).toFixed(1)}%)`,
      );
      await new Promise(requestAnimationFrame);
    }
  }
  state.syncing = false;
  state.accumulator = 0;
  state.nextClockCorrectionMs = Date.now() + CLOCK_CORRECTION_MS;
}

function frame(now) {
  if (!state.running) return;
  const rawDelta = (now - state.lastFrameTime) / 1000;
  const delta = Math.min(0.25, Math.max(0, rawDelta));
  state.lastFrameTime = now;

  if (state.observedSparseMode !== state.sparseMode) {
    state.observedSparseMode = state.sparseMode;
    resetStats(now);
    setMessage(`更新モードを${state.sparseMode ? "SPARSE" : "DENSE"}へ変更しました。`);
  }

  if (!state.paused) state.accumulator += delta * state.speed;

  if (state.nextClockCorrectionMs !== null
      && state.speed === CLOCK_GENERATIONS_PER_SECOND
      && Date.now() >= state.nextClockCorrectionMs) {
    const target = localGenerationAt(new Date());
    const difference = clockGenerationDifference(target, state.generation);
    state.accumulator = difference;
    state.nextClockCorrectionMs = Date.now() + CLOCK_CORRECTION_MS;
    const direction = difference < 0 ? "ahead" : "behind/on time";
    setMessage(`Clock correction: ${direction} by ${formatInteger(Math.abs(difference))} generations.`);
  }

  const automatic = state.paused ? 0 : Math.max(0, Math.floor(state.accumulator));
  const wanted = automatic + state.singleStep;
  const steps = Math.min(MAX_STEPS_PER_FRAME, wanted);
  const automaticUsed = Math.min(automatic, steps);
  const singleUsed = Math.min(state.singleStep, steps - automaticUsed);

  drawLife(steps);
  state.accumulator -= automaticUsed;
  state.singleStep -= singleUsed;
  state.generation += steps;
  state.statsFrameCount += 1;

  const statsElapsed = (now - state.statsStart) / 1000;
  if (statsElapsed >= 1) {
    state.measuredSpeed = (state.generation - state.statsGeneration) / statsElapsed;
    state.measuredFps = state.statsFrameCount / statsElapsed;
    state.statsStart = now;
    state.statsGeneration = state.generation;
    state.statsFrameCount = 0;
  }
  updateHud(now);
  requestAnimationFrame(frame);
}

function exposeConsoleApi() {
  const api = {
    pause() { state.paused = true; },
    run() { state.paused = false; },
    step() { state.paused = true; state.singleStep += 1; },
    reset: resetGrid,
    fullView,
  };
  Object.defineProperties(api, {
    sparseMode: {
      enumerable: true,
      get: () => state.sparseMode,
      set: (value) => { state.sparseMode = Boolean(value); },
    },
    speed: {
      enumerable: true,
      get: () => state.speed,
      set: (value) => {
        const number = Number(value);
        if (!Number.isFinite(number) || number <= 0) throw new TypeError("speed must be a positive number");
        state.speed = number;
      },
    },
    generation: { enumerable: true, get: () => state.generation },
  });
  window.life = api;
}

function showFatalError(error) {
  console.error(error);
  state.running = false;
  loading.hidden = false;
  canvas.classList.add("waiting");
  hud.hidden = true;
  controls.hidden = true;
  setLoading("起動できませんでした", `${error.message}\n\nChromeで http://localhost 経由から開いていることを確認してください。`);
}

async function main() {
  installInputHandlers();
  exposeConsoleApi();

  const syncToLocalTime = new URLSearchParams(location.search).get("sync") !== "0";
  const startupDate = new Date();
  const source = syncToLocalTime
    ? snapshotInfo(startupDate)
    : { url: "clock.rle", generation: 0, originWallMs: startupDate.getTime() };

  state.syncing = syncToLocalTime;
  state.speed = syncToLocalTime ? CLOCK_GENERATIONS_PER_SECOND : 50;
  state.initialGeneration = source.generation;
  state.generation = source.generation;
  state.syncOriginGeneration = source.generation;
  state.syncOriginWallMs = source.originWallMs;
  state.sourceName = source.url.split("/").at(-1);

  setLoading("ファイルを読み込んでいます", `${state.sourceName}\nshader.wgsl`);
  const [shaderCode, rleText] = await Promise.all([
    fetchText("shader.wgsl", "WGSL"),
    fetchText(source.url, "時計RLE").catch((error) => {
      if (syncToLocalTime) {
        throw new Error(`${error.message}\nclock-snapshot.zipをlifegame直下へ展開してください。`);
      }
      throw error;
    }),
  ]);
  if (syncToLocalTime) validateSnapshotGeneration(rleText, source.generation, source.url);

  setLoading("RLEを展開しています", `${GRID_WIDTH} x ${GRID_HEIGHT} のビット盤面を作成中です。`);
  await new Promise(requestAnimationFrame);
  const parsed = parseRle(rleText);
  state.initialWords = parsed.words;
  state.initialTiles = parsed.tiles;
  setLoading(
    "WebGPUを初期化しています",
    `${parsed.width} x ${parsed.height}, ${formatInteger(parsed.liveCount)} live cells, ${formatInteger(parsed.tiles.length)} active tiles`,
  );
  await new Promise(requestAnimationFrame);
  await createGpu(shaderCode, parsed.words, parsed.tiles);

  if (syncToLocalTime) await synchronizeClock();

  state.ready = true;
  state.lastFrameTime = performance.now();
  resetStats(state.lastFrameTime);
  canvas.classList.remove("waiting");
  loading.hidden = true;
  hud.hidden = false;
  controls.hidden = false;
  setMessage(syncToLocalTime
    ? `Local-time synchronization complete at generation ${formatInteger(state.generation)}.`
    : `${state.sourceName} を同期なしで読み込みました。`);
  updateHud();
  requestAnimationFrame(frame);
}

main().catch(showFatalError);
