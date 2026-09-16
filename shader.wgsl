// 20,000 x 20,000 Conway's Game of LifeをGPUで計算・描画するシェーダー。
//
// このファイルには2種類の仕事が同居している。
//   compute shader: 次世代のセルを計算する（step_one / step_eight）
//   render shader : 計算済みのセルを画面の色へ変える（vs_main / fs_main）
//
// 横に並ぶ32セルを、u32という32ビット整数1個へ詰める。
// たとえばビット列 000...0101 は、0番と2番のセルが生きているという意味になる。
// これにより、1回のAND・OR・XORで32セルを同時に扱える。

// -------------------- Lisp/Cから毎フレーム受け取る設定 --------------------
// vec4は4個の値をまとめた箱。GPUが読みやすい16バイト単位にそろえている。
struct Params {
    screen: vec4<f32>,       // width, height, camera zoom, unused
    camera: vec4<f32>,       // center x, center y, unused, unused
    grid: vec4<u32>,         // width, height, words/row, sparse mode (0=dense, 1=sparse)
};

// binding番号は、Cのmake_*_groupで同じ番号のGPUバッファと結び付けられる。
// readは読取専用、read_writeはGPUが結果を書き込めるという意味。
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cells_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> cells_out: array<u32>;
@group(0) @binding(3) var<storage, read> active_tiles: array<u32>;
@group(0) @binding(4) var<storage, read_write> next_flags: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> next_active_tiles: array<u32>;
@group(0) @binding(6) var<storage, read_write> next_indirect: array<atomic<u32>>;

// -------------------- タイルと共有メモリ --------------------
// 1タイルの中心は「横4 words × 縦128行」= 128 × 128セル。
// step_eightは周囲8セル分も必要なので、上下に8行、左右に1 wordの余白を付ける。
const CORE_WORDS: u32 = 4u;
const CORE_ROWS: u32 = 128u;
const BATCH_STEPS: u32 = 8u;
const TILE_WORDS: u32 = CORE_WORDS + 2u;
const TILE_ROWS: u32 = CORE_ROWS + BATCH_STEPS * 2u;
const TILE_SIZE: u32 = TILE_WORDS * TILE_ROWS;
const WORKGROUP_THREADS: u32 = 64u;
const TILE_ITERATIONS: u32 =
    (TILE_SIZE + WORKGROUP_THREADS - 1u) / WORKGROUP_THREADS;

// workgroupメモリは、同じ作業班（64スレッド）だけで共有する高速な一時置き場。
// aとbを交互に入力・出力として使うため、毎世代大きなGPUメモリへ戻さずに済む。
var<workgroup> tile_a: array<u32, 864>;
var<workgroup> tile_b: array<u32, 864>;
var<workgroup> output_alive: atomic<u32>;

fn tile_columns() -> u32 {
    return (params.grid.z + CORE_WORDS - 1u) / CORE_WORDS;
}

fn tile_rows() -> u32 {
    return (params.grid.y + CORE_ROWS - 1u) / CORE_ROWS;
}

// -------------------- 次回計算するタイルを登録する --------------------
// 生セルが残ったタイルと周囲8タイルを、次回の候補集合へ加える。
// 複数の作業班が同じ候補を同時に見つけることがあるため、atomicExchangeを使う。
// atomicは「同時に触っても途中で割り込まれない操作」で、同じ番号の二重登録を防ぐ。
fn mark_next_candidates(tile_index: u32) {
    let columns = tile_columns();
    let rows = tile_rows();
    let tile_x = i32(tile_index % columns);
    let tile_y = i32(tile_index / columns);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let x = tile_x + dx;
            let y = tile_y + dy;
            if (x >= 0 && y >= 0 && x < i32(columns) && y < i32(rows)) {
                let tile = u32(y) * columns + u32(x);
                if (atomicExchange(&next_flags[tile], 1u) == 0u) {
                    // atomicAddが返した番号を、自分専用の書込み場所として使う。
                    let slot = atomicAdd(&next_indirect[0], 1u);
                    next_active_tiles[slot] = tile;
                }
            }
        }
    }
}

fn grid_word(word_x: i32, y: i32) -> u32 {
    // 盤面の外側は常に死セル（全ビット0）として扱う。
    if (word_x < 0 || y < 0 ||
        word_x >= i32(params.grid.z) || y >= i32(params.grid.y)) {
        return 0u;
    }
    return cells_in[u32(y) * params.grid.z + u32(word_x)];
}

// -------------------- 古い出力を消す --------------------
// 2枚のセル面を交互に使うので、出力先には2世代前のセルが残っている。
// SPARSE時は全面ではなく、前回その面へ書いた候補タイルだけを0にして節約する。
// 8×8=64スレッドが、512 wordsを8個ずつ分担する。
@compute @workgroup_size(8, 8, 1)
fn clear_tiles(@builtin(workgroup_id) group: vec3<u32>,
               @builtin(local_invocation_index) local_index: u32) {
    let tile_index = active_tiles[group.x];
    let columns = tile_columns();
    let base_word = (tile_index % columns) * CORE_WORDS;
    let base_y = (tile_index / columns) * CORE_ROWS;
    var index = local_index;
    loop {
        if (index >= CORE_WORDS * CORE_ROWS) { break; }
        let word_x = base_word + index % CORE_WORDS;
        let y = base_y + index / CORE_WORDS;
        if (word_x < params.grid.z && y < params.grid.y) {
            cells_out[y * params.grid.z + word_x] = 0u;
        }
        index += 64u;
    }
}

fn life_word(al: u32, ac: u32, ar: u32,
             ml: u32, mc: u32, mr: u32,
             bl: u32, bc: u32, br: u32) -> u32 {
    // -------------------- 32セル分のLife規則を一度に計算 --------------------
    // 上・中央・下の各行について、左/中央/右wordを受け取る。
    // シフトによって各近傍セルのビット位置を、判定される中央セルへそろえる。
    // wordの端を越えた1ビットは、隣のwordから受け取る。
    let neighbours = array<u32, 8>(
        (ac << 1u) | (al >> 31u), ac, (ac >> 1u) | (ar << 31u),
        (mc << 1u) | (ml >> 31u),     (mc >> 1u) | (mr << 31u),
        (bc << 1u) | (bl >> 31u), bc, (bc >> 1u) | (br << 31u));

    // 8個の「0か1」を足し、近傍数0～8を2進数の4桁で持つ。
    // onesが1の位、twosが2の位、foursが4の位、eightsが8の位。
    // 普通の筆算の繰り上がりをAND/XORで行い、32セルを並列計算している。
    var ones = 0u;
    var twos = 0u;
    var fours = 0u;
    var eights = 0u;
    for (var i = 0u; i < 8u; i++) {
        let carry_1 = ones & neighbours[i];
        ones = ones ^ neighbours[i];
        let carry_2 = twos & carry_1;
        twos = twos ^ carry_1;
        let carry_3 = fours & carry_2;
        fours = fours ^ carry_2;
        eights = eights ^ carry_3;
    }

    // 近傍数が2なら生存中のセルだけ残り、3なら生死に関係なく誕生する（B3/S23）。
    let high_clear = ~(fours | eights);
    let exactly_two = (~ones) & twos & high_clear;
    let exactly_three = ones & twos & high_clear;
    return exactly_three | (mc & exactly_two);
}

@compute @workgroup_size(8, 8, 1)
fn step_one(@builtin(workgroup_id) group: vec3<u32>,
            @builtin(local_invocation_index) local_index: u32) {
    // -------------------- 1世代だけ進める --------------------
    // output_aliveは、このタイルに生セルが1個でもできたかを作業班全体で共有する印。
    if (local_index == 0u) { atomicStore(&output_alive, 0u); }
    workgroupBarrier();

    // DENSEならgroup.xがそのままタイル番号。SPARSEなら候補リストから番号を読む。
    let tile_index = select(group.x, active_tiles[group.x], params.grid.w != 0u);
    let columns = tile_columns();
    let base_word = (tile_index % columns) * CORE_WORDS;
    let base_y = (tile_index / columns) * CORE_ROWS;
    var index = local_index;
    loop {
        if (index >= CORE_WORDS * CORE_ROWS) { break; }
        let word_x = base_word + index % CORE_WORDS;
        let y = base_y + index / CORE_WORDS;
        if (word_x < params.grid.z && y < params.grid.y) {
            let wx = i32(word_x);
            let iy = i32(y);
            let next = life_word(
                grid_word(wx - 1, iy - 1), grid_word(wx, iy - 1), grid_word(wx + 1, iy - 1),
                grid_word(wx - 1, iy),     grid_word(wx, iy),     grid_word(wx + 1, iy),
                grid_word(wx - 1, iy + 1), grid_word(wx, iy + 1), grid_word(wx + 1, iy + 1));
            cells_out[y * params.grid.z + word_x] = next;
            if (next != 0u) { atomicStore(&output_alive, 1u); }
        }
        index += 64u;
    }
    workgroupBarrier();
    // 64スレッドのうち代表1個だけが次回候補を登録する。
    if (local_index == 0u && atomicLoad(&output_alive) != 0u) {
        mark_next_candidates(tile_index);
    }
}

// -------------------- 8世代をまとめて進める --------------------
// 128×128セルの中心に8セル幅のhalo（のりしろ）を付けて共有メモリへ読む。
// Lifeの影響は1世代に1セルしか進まないため、8世代なら8セルのhaloで足りる。
// 途中7回の結果を低速な大域メモリへ書かず、高速なtile_a/tile_b内で往復させる。
@compute @workgroup_size(8, 8, 1)
fn step_eight(@builtin(workgroup_id) group: vec3<u32>,
              @builtin(local_invocation_index) local_index: u32) {
    if (local_index == 0u) { atomicStore(&output_alive, 0u); }
    workgroupBarrier();

    let active_tile = select(group.x, active_tiles[group.x], params.grid.w != 0u);
    let columns = tile_columns();
    let base_word = i32((active_tile % columns) * CORE_WORDS);
    let base_y = i32((active_tile / columns) * CORE_ROWS);

    // 64スレッドで864 wordsを手分けして読み込む。
    // 864は64で割り切れないが、全スレッドが必ず14回反復する。
    // 最後の範囲外要素だけをifで省き、barrier前の制御フローを揃える。
    for (var iteration = 0u; iteration < TILE_ITERATIONS; iteration++) {
        let index = local_index + iteration * WORKGROUP_THREADS;
        if (index < TILE_SIZE) {
            let tx = index % TILE_WORDS;
            let ty = index / TILE_WORDS;
            tile_a[index] = grid_word(base_word + i32(tx) - 1,
                                      base_y + i32(ty) - i32(BATCH_STEPS));
        }
    }
    workgroupBarrier();

    for (var generation = 0u; generation < BATCH_STEPS; generation++) {
        // 偶数世代はa→b、奇数世代はb→a。barrierで全員の書込み完了を待つ。
        for (var iteration = 0u; iteration < TILE_ITERATIONS; iteration++) {
            let index = local_index + iteration * WORKGROUP_THREADS;
            if (index < TILE_SIZE) {
                let tx = index % TILE_WORDS;
                let ty = index / TILE_WORDS;
                let left = select(index, index - 1u, tx > 0u);
                let right = select(index, index + 1u, tx + 1u < TILE_WORDS);
                let above = select(index, index - TILE_WORDS, ty > 0u);
                let below = select(index, index + TILE_WORDS, ty + 1u < TILE_ROWS);
                let above_left = select(above, above - 1u, tx > 0u && ty > 0u);
                let above_right = select(above, above + 1u, tx + 1u < TILE_WORDS && ty > 0u);
                let below_left = select(below, below - 1u, tx > 0u && ty + 1u < TILE_ROWS);
                let below_right = select(below, below + 1u, tx + 1u < TILE_WORDS && ty + 1u < TILE_ROWS);

                let global_word = base_word + i32(tx) - 1;
                let global_y = base_y + i32(ty) - i32(BATCH_STEPS);
                let in_grid = global_word >= 0 && global_y >= 0 &&
                                global_word < i32(params.grid.z) && global_y < i32(params.grid.y);

                if ((generation & 1u) == 0u) {
                    var next = 0u;
                    if (in_grid) {
                        next = life_word(
                            tile_a[above_left], tile_a[above], tile_a[above_right],
                            tile_a[left], tile_a[index], tile_a[right],
                            tile_a[below_left], tile_a[below], tile_a[below_right]);
                    }
                    tile_b[index] = next;
                } else {
                    var next = 0u;
                    if (in_grid) {
                        next = life_word(
                            tile_b[above_left], tile_b[above], tile_b[above_right],
                            tile_b[left], tile_b[index], tile_b[right],
                            tile_b[below_left], tile_b[below], tile_b[below_right]);
                    }
                    tile_a[index] = next;
                }
            }
        }
        workgroupBarrier();
    }

    // 8は偶数なので、最後の結果はtile_aに戻っている。中心部分だけを出力面へ書く。
    var index = local_index;
    loop {
        if (index >= CORE_WORDS * CORE_ROWS) { break; }
        let core_x = index % CORE_WORDS;
        let core_y = index / CORE_WORDS;
        let word_x = u32(base_word) + core_x;
        let y = u32(base_y) + core_y;
        if (word_x < params.grid.z && y < params.grid.y) {
            let tile_index = (core_y + BATCH_STEPS) * TILE_WORDS + core_x + 1u;
            let next = tile_a[tile_index];
            cells_out[y * params.grid.z + word_x] = next;
            if (next != 0u) { atomicStore(&output_alive, 1u); }
        }
        index += 64u;
    }
    workgroupBarrier();
    if (local_index == 0u && atomicLoad(&output_alive) != 0u) {
        mark_next_candidates(active_tile);
    }
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    // -------------------- 画面全体を覆う三角形 --------------------
    // 頂点3個の巨大な三角形で画面を覆う。各ピクセルの色はfs_mainが決める。
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(positions[index], 0.0, 1.0);
}

fn range_has_life(x0: i32, y0: i32, x1: i32, y1: i32) -> bool {
    // 指定した長方形内に生セルが1個でもあるかを調べる。
    // 縮小表示では多数のセルが1ピクセルへ入るので、どれかが生なら点を残す。
    let clipped_x0 = clamp(x0, 0, i32(params.grid.x) - 1);
    let clipped_x1 = clamp(x1, 0, i32(params.grid.x) - 1);
    let clipped_y0 = clamp(y0, 0, i32(params.grid.y) - 1);
    let clipped_y1 = clamp(y1, 0, i32(params.grid.y) - 1);
    if (clipped_x0 > clipped_x1 || clipped_y0 > clipped_y1) { return false; }

    let first_word = u32(clipped_x0) >> 5u;
    let last_word = u32(clipped_x1) >> 5u;
    let first_bit = u32(clipped_x0) & 31u;
    let last_bit = u32(clipped_x1) & 31u;
    let first_mask = 0xffffffffu << first_bit;
    let last_mask = select((1u << (last_bit + 1u)) - 1u, 0xffffffffu, last_bit == 31u);

    // zoom上限は64 cells/pixel。丸めの余白込みでも最大66行、横4 wordsを見ればよい。
    for (var row_offset = 0u; row_offset < 66u; row_offset++) {
        let y = u32(clipped_y0) + row_offset;
        if (y > u32(clipped_y1)) { break; }
        for (var word_offset = 0u; word_offset < 4u; word_offset++) {
            let word_x = first_word + word_offset;
            if (word_x > last_word) { break; }
            var mask = 0xffffffffu;
            if (word_x == first_word) { mask = mask & first_mask; }
            if (word_x == last_word) { mask = mask & last_mask; }
            if ((cells_in[y * params.grid.z + word_x] & mask) != 0u) { return true; }
        }
    }
    return false;
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // -------------------- 各画面ピクセルの色を決める --------------------
    let zoom = params.screen.z;
    // 画面座標positionを、盤面上のセル座標worldへ変換する。
    let world = params.camera.xy + (position.xy - params.screen.xy * 0.5) * zoom;
    // 正確な盤面境界の両側へ画面0.5pxずつ広げ、合計1pxの青線にする。
    let border_width = max(zoom * 0.5, 0.025);
    let along_horizontal_edge = world.x >= -border_width &&
                                world.x <= f32(params.grid.x) + border_width &&
                                (abs(world.y) <= border_width ||
                                 abs(world.y - f32(params.grid.y)) <= border_width);
    let along_vertical_edge = world.y >= -border_width &&
                              world.y <= f32(params.grid.y) + border_width &&
                              (abs(world.x) <= border_width ||
                               abs(world.x - f32(params.grid.x)) <= border_width);
    if (along_horizontal_edge || along_vertical_edge) {
        return vec4<f32>(0.08, 0.43, 1.00, 1.0);
    }
    let outside = world.x < 0.0 || world.y < 0.0 ||
                  world.x >= f32(params.grid.x) || world.y >= f32(params.grid.y);
    if (outside) {
        return vec4<f32>(0.008, 0.011, 0.015, 1.0);
    }

    var alive = false;
    var grid_line = 0.0;
    if (zoom < 1.0) {
        // 拡大時: 1セルが複数ピクセルになる。セル境界を薄い格子線として加える。
        let cell = vec2<i32>(floor(world));
        alive = range_has_life(cell.x, cell.y, cell.x, cell.y);
        let local = fract(world);
        let edge = min(min(local.x, 1.0 - local.x), min(local.y, 1.0 - local.y));
        grid_line = 1.0 - smoothstep(0.0, min(0.08 / zoom, 0.22), edge);
    } else {
        // 縮小時: 1ピクセルが覆うセル範囲を調べ、細い配線が消えないようORを取る。
        let half_span = zoom * 0.5;
        alive = range_has_life(i32(floor(world.x - half_span)),
                               i32(floor(world.y - half_span)),
                               i32(floor(world.x + half_span)),
                               i32(floor(world.y + half_span)));
    }

    // 最後に「背景色」か「生セルの灰色」を選び、alpha=1（不透明）で返す。
    let background = vec3<f32>(0.012, 0.020, 0.027) + vec3<f32>(0.010, 0.018, 0.022) * grid_line;
    let live_color = vec3<f32>(0.78, 0.81, 0.84);
    return vec4<f32>(select(background, live_color, alive), 1.0);
}
