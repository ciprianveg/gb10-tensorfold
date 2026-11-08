//! glm53r2's native registration (§118): the standard OpenAI surface over the lane host's LONE
//! driver — our whole-request Session.exec under the §98 rendezvous (rank 0 broadcasts the header +
//! the prompt ids; every follower runs Session.exec alike, the NCCL collectives keeping the ranks in
//! lockstep). The v1 limits, named: ONE drafted stream at a time (the lane core's shared rounds —
//! the multi-window verify over our cuda_multi — are the v2 seam; concurrent requests wait for the
//! lone stream to finish); think budgets, loop guards, structured output and logprobs route to the
//! lane core and refuse by name. The sampling flows lanes.Sampling -> smp.Sampling -> the walk's
//! Gumbel (dspark_noise 0.7) — the sampled class served natively for the first time.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const rn = @import("cuda_runner.zig");
const smp = @import("sampling.zig");
const fw = @import("cuda_forward.zig");
const seam = @import("cuda_lanes.zig");
const multi = @import("cuda_multi.zig");
const mdraft = @import("cuda_mdraft.zig");
const core = @import("core");

pub const model_type = "glm_moe_dsa";
pub const formats: []const []const u8 = &.{"modelopt-nvfp4"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = 4096; // the served prompt chunk (the launcher PROMPT_ROWS shape)
pub const explains_open = true;

/// --context maps the engine's own limit; every other setting rides the family's env surface
/// (TF_GLM53_*: the launchers' pins — kv8/dcp/splitk/dspark-conf — exactly as the hand-rolled serve).
pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
};

/// native/cuda.zig's LoneRun, mirrored: the lane host calls it with the family's ctx on its engine thread.
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    /// The family's own pool declarations (the multi form): the DCP2 planes pre-allocate inside the
    /// model's bytes (no marginal stream cost — the budget must not double-count them), the family's
    /// admission governs (the aggregate cap), and the prompts are the pool's own keeping (§128.27cj).
    layout: ?core.kv_pool.Layout = null,
    engine_paged: bool = false,
    own_prompt_cache: bool = false,
};

/// §98's request header (the serve lane's form): rank 0 broadcasts it then the prompt ids; a
/// header with a foreign magic is the followers' shutdown signal.
const hdr_magic: u32 = 0x52334747;
const Hdr = extern struct {
    magic: u32,
    n_prompt: u32,
    max_tokens: u32,
    draft: u32,
    // 118.4: the sampling rides the rendezvous (temp 0: greedy, the anchors' path). The f64s as raw bits.
    seed: u64 = 0,
    temp: u64 = 0,
    top_p: u64 = 0,
    min_p: u64 = 0,
    top_k: u32 = 0,
    pad: u32 = 0,

    fn sampling(h: *const Hdr) ?smp.Sampling {
        if (@as(f64, @bitCast(h.temp)) <= 0) return null;
        return .{
            .seed = h.seed,
            .temperature = @bitCast(h.temp),
            .top_k = h.top_k,
            .top_p = @bitCast(h.top_p),
            .min_p = @bitCast(h.min_p),
        };
    }

    fn fill(h: *Hdr, s: ?lanes.Sampling) void {
        const x = s orelse {
            h.seed = 0;
            h.temp = 0;
            h.top_p = 0;
            h.min_p = 0;
            h.top_k = 0;
            return;
        };
        h.seed = x.seed;
        h.temp = @bitCast(x.temperature);
        h.top_p = @bitCast(x.top_p);
        h.min_p = @bitCast(x.min_p);
        h.top_k = x.top_k;
    }
};

const Owned = struct {
    gpa: std.mem.Allocator,
    boot: *engine.Boot,
    session: *engine.Session,
    adapter: ?*Adapter = null,
};

/// 118.5: TF_GLM53_ENGINE_CPU=<n> pins the engine thread to CPU n; TF_GLM53_ENGINE_FIFO=<prio> asks for
/// SCHED_FIFO (needs CAP_SYS_NICE; a refusal is logged so a cap-miss cannot read as a fix-failure). The
/// exchange trace showed rank 0 - the server rank - arriving last at 63% of exchanges (median wait
/// 172.7us vs the peers' 141.2, the fat-tail gaps the framework threads' preemption); the proxy pin was
/// null because the late party is the ENGINE thread. Applied once, on the thread that runs the rounds.
var sched_done = false;
const sched_c = struct {
    const Param = extern struct { sched_priority: c_int };
    extern "c" fn sched_setaffinity(pid: c_int, cpusetsize: usize, mask: *const u64) c_int;
    extern "c" fn sched_setscheduler(pid: c_int, policy: c_int, param: *const Param) c_int;
};
fn envNum(comptime name: [:0]const u8) ?i32 {
    const v = std.c.getenv(name) orelse return null;
    return std.fmt.parseInt(i32, std.mem.span(v), 10) catch null;
}
fn engineSched() void {
    if (sched_done) return;
    sched_done = true;
    if (envNum("TF_GLM53_ENGINE_CPU")) |n| {
        const set: u64 = @as(u64, 1) << @intCast(@as(u6, @intCast(@mod(n, 64))));
        if (sched_c.sched_setaffinity(0, @sizeOf(u64), &set) != 0)
            std.log.warn("engine thread: sched_setaffinity(cpu {d}) refused", .{n})
        else
            std.log.info("engine thread: pinned to CPU {d}", .{n});
    }
    if (envNum("TF_GLM53_ENGINE_FIFO")) |prio| {
        const sp = sched_c.Param{ .sched_priority = prio };
        if (sched_c.sched_setscheduler(0, 1, &sp) != 0) // 1: SCHED_FIFO
            std.log.warn("engine thread: SCHED_FIFO at {d} refused (CAP_SYS_NICE?)", .{prio})
        else
            std.log.info("engine thread: SCHED_FIFO at {d}", .{prio});
    }
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = ctx; // the Boot opens its own driver from the env (Options.fromEnv's lane)
    _ = kernels; // the captured sets are the engine's own surface (TF_GLM53_AOT)
    const opts = try engine.Options.fromEnv(dir, o.context);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    const boot = try engine.Boot.init(gpa, io, opts);
    const session = try gpa.create(engine.Session);
    session.* = try engine.Session.init(gpa, boot);
    own.* = .{ .gpa = gpa, .boot = boot, .session = session };
    // THE V2 SEAM'S SWAP (§128.27ai): with concurrent slots (Boot.multi at --parallel > 1) the lane core's
    // Backend is the adapter over the multi's windows — the shared rounds the registration named; the lone
    // driver and its stub vtable stay the --parallel 1 form. The stream_bytes: the multi's own per-stream
    // measure (the halved DCP2 plane included).
    var adapter: ?*Adapter = null;
    if (boot.multi) |mu| {
        const ad = try gpa.create(Adapter);
        ad.* = Adapter.init(gpa, mu, &boot.rdv, boot.io);
        ad.mode = if (boot.o.mode) |m| multi.modeOf(std.meta.stringToEnum(rn.Mode, m) orelse .dspark) else .dspark; // §128.27ew the engine's draft mode (the env string parsed; dspark the unset default, the prior behavior)
        adapter = ad;
        own.adapter = ad;
        if (opts.rank == 0 and boot.rdv.world > 1) {
            if (std.Thread.spawn(.{}, heartbeatLoop, .{ad})) |t| t.detach() else |_| {}
        }
    }
    if (opts.rank != 0) { // a follower never serves: the §98 loop until the shutdown header
        const fv2 = if (boot.multi != null) followerV2(own) else follower(own);
        fv2 catch |e| {
            std.log.info("rank {d}: following rank 0 ended ({s})", .{ opts.rank, @errorName(e) });
        };
        std.process.exit(0);
    }
    return .{
        .backend = if (adapter) |ad| .{ .ptr = ad, .vtable = &adapter_vtable } else .{ .ptr = own, .vtable = &vtable },
        .facts = if (boot.multi != null)
            .{ .mtp = true, .drafts = @as(u32, 7), .exact_width = @intCast(fw.decode_rows), .copy_drafts = false, .speculate = true, .speculate_early = false, .hidden_rows = true, .batch_rows = @intCast(fw.decode_rows), .max_streams = multi.max_streams,
            .draft_prior = &.{ 0.634, 0.495, 0.438, 0.387, 0.300, 0.250 }, // 128.27fn THE TRUE MARGINALS (128.27fo: the dlog s kept/tried, the fit tool s own semantics: d0-d3 the raw 170/268, 93/188, 39/89, 12/31; d4-d5 the small-sample-conservative trend) - the Rule s priced search runs on the measured profile, not the class defaults (the LEARN-WITHOUT-FEED class s static remedy; the live observe-wiring priced behind)
            .window_costs = &.{
                .{ .width = 1, .ms = 78.0 }, .{ .width = 2, .ms = 86.0 }, .{ .width = 3, .ms = 88.0 },
                .{ .width = 4, .ms = 89.0 }, .{ .width = 5, .ms = 90.0 }, .{ .width = 6, .ms = 91.0 },
                .{ .width = 7, .ms = 92.0 }, .{ .width = 8, .ms = 93.0 },
            } }
        else
            .{ .mtp = true, .drafts = @intCast(opts.verify_rows), .exact_width = 1, .copy_drafts = false, .speculate = false, .speculate_early = false },
        .rows = @intCast(fw.decode_rows),
        .free_streams = if (boot.multi != null) @intCast(boot.o.parallel) else 1,
        .layout = if (boot.multi != null) .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = 4096, .snap_bytes = 0 } else null,
        .engine_paged = boot.multi != null,
        .own_prompt_cache = boot.multi != null,
        .stream_bytes = if (boot.multi) |mu| mu.bytesPerStream() else opts.context * 51712, // the kv8 plane at DCP1: ~50.6 KiB/token/rank (findings-kv8.md; the §109 b(1) receipt 1.59 GiB at 32K); the multi's own measure at --parallel > 1
        .ctx = own,
        .deinit = close,
        // The lone's §98 wire and the adapter's V2 ops share the follower's one loop and the NCCL
    // beneath: run concurrently (the host's lone-first setup beside the engine's rounds) their
    // collectives cross — the lone's exec wedges, the adapter's rounds compute over wrong
    // allreduces (the degenerate pair receipt). On the multi form every stream goes through the
    // adapter; the solo boots keep the lone.
    .lone = if (boot.multi != null) null else loneRun,
    };
}

/// The followers' §98 loop: the rendezvous broadcast, then Session.exec alike on every rank.
/// THE V2 SEAM'S RANK COORDINATION (§128.27ak): the Link form — rank 0's adapter broadcasts each
/// operation as it happens (admit + the prompt, fill(sid), round(the plan words), finish(sid)); the
/// followers apply the SAME sequence to their own Multi (applyFill / applyRound — the Multi's own
/// follower path, the out-of-step guard backstopping). The choice receipted: the request-mirror's
/// step-loop form risks the fill/round interleaving divergence (the adapter's whole-prompt prefill vs
/// the step's one-chunk-a-round); the Link pins the sequence. The v1's zero-magic shutdown header
/// already terminates this loop (the magic mismatch returns).
pub const v2_magic = seam.v2_magic;
pub const V2Kind = seam.V2Kind;
pub const V2Msg = seam.V2Msg;

/// the admit's mirrored fields (the sampling included — the follower's picks draw with it).
const V2Admit = seam.V2Admit;

/// §128.27hz+++++++++++++++++++ THE IDLE HEARTBEAT LOOP (the leader only): every 240 s, a no-op
/// heartbeat op under the wire mutex (an op-suite like any other - never two threads inside). The
/// followers' dispatch ignores it and their recvAll restarts with a fresh 600 s: a healthy idle
/// leader keeps its workers forever, a dead one still frees them within 10 min (the timeout's
/// original purpose intact). A heartbeat send that fails (the followers already gone) ends the
/// thread with the error named - the next request surfaces the death as before.
fn heartbeatLoop(d: *Adapter) void {
    while (true) {
        std.Io.sleep(d.io, .fromMilliseconds(240_000), .awake) catch {};
        d.wire.lockUncancelable(d.io);
        defer d.wire.unlock(d.io);
        d.sendRaw(.heartbeat, 0, 0) catch |e| {
            std.log.err("adapter: the idle heartbeat failed (the followers gone?): {s}", .{@errorName(e)});
            return;
        };
    }
}

/// THE UNIFIED FOLLOWER (the boundary-6 fix): the lane host uses BOTH paths — the lone driver for
/// single requests (the warm-up, the serial anchors) and the Engine's rounds for the concurrent — so
/// the follower must serve both wires, message by message. ONE receive of the Hdr-sized header; the
/// v2 magic dispatches the Link, the §98 magic the lone exec, anything else the shutdown.
fn followerV2(own: *Owned) !void {
    engineSched();
    const a = own.gpa;
    const mu = own.boot.multi orelse return error.NoMulti;
    while (true) {
        var hdr: Hdr = undefined;
        try own.boot.rdv.broadcast(std.mem.asBytes(&hdr));
        if (hdr.magic == v2_magic) {
            const msg: *const V2Msg = @ptrCast(&hdr); // the extern-struct reinterpretation (the same 56 bytes)
            switch (@as(V2Kind, @enumFromInt(msg.kind))) {
                .shutdown => return,
                .keep => return error.OutOfStep, // a keep never arrives at the loop's top: only inside a round
                .finish => mu.finish(msg.sid),
                .fill => _ = mu.applyFill(msg.sid) catch |e| {
                    std.log.err("follow: fill sid={d} failed at rank {d}: {s}", .{ msg.sid, mu.w.rank, @errorName(e) });
                    return e;
                },
                .admit => {
                    const ids = try a.alloc(u32, msg.n);
                    defer a.free(ids);
                    try own.boot.rdv.broadcast(std.mem.sliceAsBytes(ids));
                    var aw: V2Admit = undefined;
                    try own.boot.rdv.broadcast(std.mem.asBytes(&aw));
                    var gmask: ?[]u32 = null;
                    defer if (gmask) |m| a.free(m);
                    if (aw.has_grammar != 0) {
                        gmask = try a.alloc(u32, aw.grammar_words);
                        try own.boot.rdv.broadcast(std.mem.sliceAsBytes(gmask.?));
                    }
                    const sampling: ?smp.Sampling = if (aw.has_sampling != 0) .{
                        .seed = aw.seed, .temperature = aw.temperature, .top_k = aw.top_k, .top_p = aw.top_p, .min_p = aw.min_p,
                    } else null;
                    var stops_list: std.ArrayList(usize) = .empty;
                    defer stops_list.deinit(a);
                    {
                        const pts = try mu.plan_rz.points(a, ids);
                        defer a.free(pts);
                        for (pts) |p| if (aw.begin < p and p < ids.len) try stops_list.append(a, p);
                    }
                    _ = mu.admit(.{ .sid = msg.sid, .slot = aw.slot, .mode = @enumFromInt(aw.mode), .sampling = sampling, .count = aw.count, .prompt = ids,
                        .begin = aw.begin, .stops = stops_list.items, .keeps = aw.keeps[0..@min(aw.keeps_n, 4)],
                        .gmask = gmask,
                        .src = if (aw.has_src != 0) aw.src else null }) catch |e| switch (e) {
                        error.PoolFull, error.NoFreeSlot, error.PromptTooLong => {},
                        else => return e,
                    };
                },
                .round => {
                    const words = try a.alloc(u32, msg.n);
                    defer a.free(words);
                    try own.boot.rdv.broadcast(std.mem.sliceAsBytes(words));
                    const plan = try multi.Plan.decode(words);
                    {
                        var gs: [multi.max_streams]u32 = @splat(0);
                        for (plan.items[0..plan.n], 0..) |it, i| gs[i] = it.grammar;
                        std.log.info("follower: round-entry n={d} grammars={any}", .{ plan.n, gs[0..plan.n] });
                    }
                    var t = try mu.gpu(&plan);
                    std.log.info("follower: post-gpu", .{}); // §128.27hz the wedge hunt's stage marks
                    var fres: [multi.max_streams]multi.Result = undefined;
                    try mu.picks(&plan, &t, &fres, null);
                    std.log.info("follower: post-picks", .{}); // §128.27hz
                    try mu.maskPicks(&plan, &t, &fres, null);
                    std.log.info("follower: post-maskPicks", .{}); // §128.27hz the wedge hunt's stage marks
                    try mu.commit(&plan, &t, &fres);
                    std.log.info("follower: post-commit", .{}); // §128.27hz
                    for (plan.items[0..plan.n]) |it| {
                        const s2 = mu.find(it.sid) orelse return error.OutOfStep;
                        s2.P = it.P;
                        s2.tok = it.tok;
                        s2.m = it.m;
                    }
                },
                .heartbeat => {},
            }
            continue;
        }
        if (hdr.magic != hdr_magic) return; // the shutdown signal (or corruption): leave
        // the §98 lone wire: the prompt ids then Session.exec (the v1 path, unchanged)
        const ids = try a.alloc(u32, hdr.n_prompt);
        defer a.free(ids);
        try own.boot.rdv.broadcast(std.mem.sliceAsBytes(ids));
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(a);
        _ = try own.session.exec(.{ .prompt = ids, .max_tokens = hdr.max_tokens, .draft = hdr.draft != 0, .eos = own.session.eos, .copies = own.boot.o.copy.on, .sampling = hdr.sampling() }, &out);
    }
}

fn follower(own: *Owned) !void {
    engineSched(); // the followers' engine thread (this process's main)
    const a = own.gpa;
    while (true) {
        var hdr: Hdr = undefined;
        try own.boot.rdv.broadcast(std.mem.asBytes(&hdr));
        if (hdr.magic != hdr_magic) return; // the shutdown signal
        const ids = try a.alloc(u32, hdr.n_prompt);
        defer a.free(ids);
        try own.boot.rdv.broadcast(std.mem.sliceAsBytes(ids));
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(a);
        _ = try own.session.exec(.{ .prompt = ids, .max_tokens = hdr.max_tokens, .draft = hdr.draft != 0, .eos = own.session.eos, .copies = own.boot.o.copy.on, .sampling = hdr.sampling() }, &out);
    }
}

/// The lone driver's hook state (one engine thread, one lone stream at a time).
const HookState = struct {
    own: *Owned,
    stream: *lanes.Stream,
    committed: *const fn (*anyopaque) void,
    yield: *const fn (*anyopaque) bool,
    hooks_ctx: *anyopaque,
    failed: ?anyerror = null,
};

/// rn.Hooks.tokens: the round's tokens commit to the lane stream (the first pick alone, then each
/// round's — the Runner calls this for every commit in order); the lane host's yield (a foreground
/// waiter, a cancel, a shutdown) ends the stream cleanly through the StopVote.
fn hookTokens(ctx: *anyopaque, toks: []const u32) bool {
    const hs: *HookState = @ptrCast(@alignCast(ctx));
    _ = hs.stream.commit(hs.own.gpa, toks, &.{}) catch |e| {
        hs.failed = e;
        return true;
    };
    hs.committed(hs.hooks_ctx);
    return hs.yield(hs.hooks_ctx);
}

/// A lone drafted stream, driven as the hand-rolled serve drives one request: the §98 header + prompt
/// broadcast, then Session.exec with the stream's sampling (the Gumbel's road) and the commit hook.
/// Never hands back mid-round (resumed is refused): a yield ends the stream instead — the v2 resume
/// (the drafts.Rounds.resumed class) is the named follow-up.
fn loneRun(p: *anyopaque, s: *lanes.Stream, hooks_p: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool {
    const own: *Owned = @ptrCast(@alignCast(p));
    engineSched(); // the server rank's engine thread (the lane host's)
    if (resumed) return true; // never handed over, so a resume is the core keeping the stream
    var hs: HookState = .{ .own = own, .stream = s, .committed = committed, .yield = yield, .hooks_ctx = hooks_p };
    var hdr = Hdr{ .magic = hdr_magic, .n_prompt = @intCast(s.prompt().len), .max_tokens = s.max_new, .draft = 1 };
    hdr.fill(s.sampling);
    const ids = try own.gpa.alloc(u32, s.prompt().len); // the broadcast wants mutable bytes
    defer own.gpa.free(ids);
    @memcpy(ids, s.prompt());
    try own.boot.rdv.broadcast(std.mem.asBytes(&hdr));
    try own.boot.rdv.broadcast(std.mem.sliceAsBytes(ids));
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(own.gpa);
    const fin = try own.session.exec(.{
        .prompt = ids,
        .max_tokens = s.max_new,
        .draft = true,
        .eos = own.session.eos,
        .copies = own.boot.o.copy.on,
        .sampling = toOur(s.sampling),
        .hooks = .{ .ctx = @ptrCast(&hs), .tokens = hookTokens },
    }, &out);
    // §125: the lone path's stats — the server's `speculative` reads the stream's counters; the
    // exec's Fin carries the runner's (the §98 serve's own source). Without this every lone request
    // reported rounds=0 (the §123.3 reporting gap).
    s.rounds = fin.stats.rounds;
    s.drafted = fin.stats.drafted;
    s.accepted = fin.stats.accepted;
    if (hs.failed) |e| return e;
    return false; // the stream ran to its end
}

/// lanes.Sampling -> smp.Sampling (field-identical; top_k u32 -> usize).
fn toOur(s: ?lanes.Sampling) ?smp.Sampling {
    const x = s orelse return null;
    if (x.temperature <= 0) return null; // temperature 0: greedy, the anchors' path
    return .{ .seed = x.seed, .temperature = x.temperature, .top_k = @intCast(x.top_k), .top_p = x.top_p, .min_p = x.min_p };
}

/// §128.27cw THE RETIREMENT: the MultiDriver — the family's own Link loop handed to the lane host
/// as the Core vtable — is RETIRED. Nine boundaries of instrumentation exonerated its logic (the
/// wire, the lock, the mdraft, the mailbox, the emit, the store all clean); its CLASS — the family
/// owning the orchestration, the lane/job ownership split across two authorities — was the wedge's
/// (the snd capture: the lane stream at 23 emitted/finished=false while the owner's lane reaped;
/// upstream's #1438, the discard-before-finish lifecycle, is the framework's own fix). The v1.2.1
/// Engine owns the orchestration; the multi form rides it through the Backend vtable (the adapter,
/// the next landing); this build serves the lone path (the §98 rendezvous, the anchors' road).

// -- §128.27cy THE ADAPTER: the Multi as the framework Engine's Backend -------------------------
//
// The Engine owns the orchestration (the #1438 discard-before-finish lifecycle, the rounds, the
// windows, the conclude); the family serves the ops. The DSpark drafting maps to the Engine's
// draft-for-next-round protocol: draft() proposes from the mdraft's committed taps and holds the
// count on the lane stream (Held); the next round's window carries those rows and verify() runs the
// Multi's own fused forward (the drafter re-proposes identically from the same taps — deterministic,
// the room capped so the gpu's own depth matches the held count; a mismatch errors loudly, the
// out-of-step class). keep() is the no-op: applyRound's commit IS the keep, and the no-keep wire
// rides beneath — the picks are a collective, every rank commits its own, the keep's broadcast dead
// by design. THE POOL owns the admission (choose/admit at prefill, finish at release); the intervals
// and the no-keep wire untouched.

const Adapter = struct {
    gpa: std.mem.Allocator,
    mu: *multi.Multi,
    mode: multi.Mode = .dspark,
    rdv: ?*cuda.rendezvous.Rendezvous = null,
    owners: [multi.max_streams]?*AOwner = @splat(null),
    next_sid: u32 = 1,
    /// §128.27hz++++++++++++++++++++++++++++++++ THE PORT'S BUFFER (the Invalid hunt's verdict,
    /// named by the numbers instrument: buf_words=165 at the first POST-THINKING masked round —
    /// the thinking phase runs the walk inactive (next() null behind the after gate, need=15
    /// fits), the first true mask (4840 words) hit the old items-only bound. THIS Adapter's
    /// buffer is the wire's (cuda_lanes' own Adapter is the seam's, a different struct).
    plan_words_buf: [multi.plan_words + multi.grammar_pool_words]u32 = @splat(0),
    res_buf: [multi.max_streams]multi.Result = undefined,
    ds_buf: [multi.max_streams]usize = undefined,
    rows_buf: [multi.max_streams][fw.decode_rows]u32 = undefined,
    held_buf: [multi.max_streams]lanes.stream.Held = undefined,
    /// THE WIRE MUTEX: the V2 ops and the NCCL collectives beneath them are order-critical
    /// cross-rank. The host's setup runs prefills on request threads beside the engine loop's
    /// rounds, so every op-suite takes the lock — the Multi's native interleave (a fill chunk, a
    /// round, a fill chunk) stays safe at op granularity; never two threads inside.
    wire: std.Io.Mutex = .init,
    io: std.Io,

    const AOwner = struct {
        lane: *lanes.Stream,
        sid: u32,
        slot: usize,
        held_i: usize = 0, // the owners-array index (the slot)
    };

    fn init(gpa: std.mem.Allocator, mu: *multi.Multi, rdv: ?*cuda.rendezvous.Rendezvous, io: std.Io) Adapter {
        return .{ .gpa = gpa, .mu = mu, .rdv = rdv, .io = io };
    }

    fn sendRaw(d: *Adapter, kind: seam.V2Kind, sid: u32, n: u32) !void {
        const rdv = d.rdv orelse return;
        var msg = seam.V2Msg{ .magic = v2_magic, .kind = @intFromEnum(kind), .sid = sid, .n = n };
        try rdv.broadcast(std.mem.asBytes(&msg));
    }

    fn ownerOfLane(d: *Adapter, s: *lanes.Stream) ?*AOwner {
        for (d.owners) |x| if (x) |o| {
            if (o.lane == s) return o;
        };
        return null;
    }

    fn freeOwner(d: *Adapter, o: *AOwner) void {
        d.owners[o.held_i] = null;
        d.gpa.destroy(o);
    }

    fn fini(d: *Adapter) void {
        for (d.owners) |x| if (x) |o| d.gpa.destroy(o);
    }

    /// The Multi's own depth formula (the same caps mu.step applied) — the count the gpu's drafter
    /// will produce for this stream, so the held count and the verify window agree by construction.
    fn depthFor(d: *Adapter, o: *AOwner) usize {
        const mu = d.mu;
        const s = mu.find(o.sid) orelse return 0;
        const which = multi.drafterOf(s.mode) orelse return 0;
        var n_live: usize = 0;
        for (mu.streams.items) |x| n_live += @intFromBool(!x.done);
        return @min(mu.depthOf(which, n_live), @min(s.count -| s.out.items.len -| 1, mu.r.capacity -| s.P -| 1));
    }

    /// draft()'s worker: the standalone propose over the committed taps, the count held on the lane.
    /// The count-hold: the held rows the Engine's next window sizes itself on. THE PROPOSALS ARE THE
    /// GPU'S OWN — its fused propose drafts the twins inside the collective round (every rank runs
    /// the same gpu over the .round op). The standalone md.propose this replaced was a rank-0-only
    /// call into a sharded head (deviceBytes/N), its all-reduce meeting no follower: the first-serve
    /// hang. The verify maps the twins from the Tables the gpu returns (the window's rows in vids).
    fn holdCount(d: *Adapter, o: *AOwner) !void {
        const depth = d.depthFor(o);
        if (depth == 0) return;
        d.held_buf[o.held_i] = .{ .count = @intCast(depth) };
        o.lane.next = d.held_buf[o.held_i];
    }

    /// The Link the Multi's own lead path drives: the fill and round ops on the wire. The 8-rank wire
    /// has no upstream analogue, but the ORDER is the Multi's (leadFill: the op before the collective)
    /// — delegated, not re-derived.
    fn linkFill(ctx: *anyopaque, sid: u32) anyerror!void {
        const d: *Adapter = @ptrCast(@alignCast(ctx));
        try d.sendRaw(.fill, sid, 0);
    }

    fn linkRound(ctx: *anyopaque, plan: *const multi.Plan) anyerror!void {
        const d: *Adapter = @ptrCast(@alignCast(ctx));
        const rdv = d.rdv orelse return;
        const words = try plan.encode(&d.plan_words_buf);
        try d.sendRaw(.round, 0, @intCast(words.len));
        try rdv.broadcast(std.mem.sliceAsBytes(words));
    }

    // -- the Backend vtable's surface ---------------------------------------------------------

    fn aPrefill(p: *anyopaque, s: *lanes.Stream) anyerror!void {
        fw.wdebug_on = false; // 128.27fq THE WDEBUG FLOOD RETIRED FROM THE SERVE PATH: the print-bisect s unconditional true taxed EVERY round s host path (the propose/ask/drafted/plane-check/fargs lines) - the shrank GPU work at the shallow ask exposed a ~60-70ms host floor (the c1 s 82ms wall at ask 2 vs the era s 16.4 flood-free); the flag stays for the instrumented boots (set in code, never at serve)
        std.log.info("adapter: prefill-entry", .{});
        const d: *Adapter = @ptrCast(@alignCast(p));
        const ids = s.prompt();
        if (ids.len == 0) return error.PromptTooLong;
        if (s.grammar != null and seam.toOur(s.sampling) != null) return error.StructuresNeedGreedy;
        // THE WHOLE PREFILL UNDER THE WIRE MUTEX (the Flash Next sid-race form: the lock BEFORE
        // next_sid is read) — and the fill's queue-head guarantee in one: no other prefill can take
        // the head between our chunks, so the Multi's own leadFill always fills OUR stream (the
        // chunk interleave deferred to the receipts' era).
        d.wire.lockUncancelable(d.io);
        defer d.wire.unlock(d.io);
        const c = try d.mu.choose(ids, true); // the SlotReuse + the pool's admission
        defer c.deinit(d.gpa);
        const o = try d.gpa.create(AOwner);
        errdefer d.gpa.destroy(o);
        const sid = d.next_sid;
        const sm = toOur(s.sampling);
        const a_mode: multi.Mode = if (s.drafts and s.grammar == null) d.mode else .serial; // §128.27ew the routed mode (was: hardcoded .dspark)
        // ...AND THE LANE-SIDE TWIN: the ENGINE must hold no drafts either — windows.zig's
        // constrain() reads the held drafts through backend.held (the v2 seam, unimplemented
        // here) the moment p.held > 0. Drafts off on the engine's own stream: the window one
        // row, the grammar's walk the pending alone, the HeldDraftsUnreadable class dead.
        if (s.grammar != null) s.drafts = false;
        const a_count: usize = @max(1, s.max_new);
        var gmask0: ?[]u32 = null;
        defer if (gmask0) |m| d.gpa.free(m);
        if (s.grammar) |g| gmask0 = try g.next(d.gpa);
        // THE WIRE: the header, the prompt, the fields — then the local admit (a local refusal after
        // the wire marks the world broken; the followers took the admission).
        try d.sendRaw(.admit, sid, @intCast(ids.len));
        if (d.rdv) |rdv| {
            const wire = try d.gpa.dupe(u32, ids);
            defer d.gpa.free(wire);
            try rdv.broadcast(std.mem.sliceAsBytes(wire));
            var aw = seam.V2Admit{
                .slot = @intCast(c.slot),
                .mode = @intFromEnum(a_mode),
                .count = @intCast(a_count),
                .has_sampling = @intFromBool(sm != null),
                .seed = if (sm) |x| x.seed else 0,
                .temperature = if (sm) |x| @floatCast(x.temperature) else 0,
                .top_k = if (sm) |x| @intCast(x.top_k) else 0,
                .top_p = if (sm) |x| @floatCast(x.top_p) else 0,
                .min_p = if (sm) |x| @floatCast(x.min_p) else 0,
                .begin = @intCast(c.begin),
                .keeps_n = @intCast(@min(c.keeps.len, 4)),
                .keeps = blk: {
                    var buf: [4]u32 = @splat(0);
                    for (c.keeps[0..@min(c.keeps.len, 4)], 0..) |k, i| buf[i] = k;
                    break :blk buf;
                },
                .has_src = @intFromBool(c.src != null),
                .src = @intCast(c.src orelse 0),
                .has_grammar = @intFromBool(gmask0 != null),
                .grammar_words = @intCast(if (gmask0) |m| m.len else 0),
            };
            try rdv.broadcast(std.mem.asBytes(&aw));
            if (gmask0) |m| try rdv.broadcast(std.mem.sliceAsBytes(m));
        }
        o.* = .{ .lane = s, .sid = sid, .slot = c.slot, .held_i = c.slot };
        _ = d.mu.admit(.{ .sid = sid, .slot = c.slot, .mode = a_mode, .sampling = sm, .count = a_count, .prompt = ids, .eos = s.eos,
                .begin = c.begin, .stops = c.stops, .keeps = c.keeps, .src = c.src,
                .gmask = gmask0 }) catch |e| switch (e) {
            error.PoolFull, error.NoFreeSlot, error.PromptTooLong => return e, // the errdefer frees the owner; the reply path surfaces the refusal
            else => {
                d.mu.broken = d.mu.world > 1;
                return e; // the errdefer frees the owner (a manual destroy here double-freed it)
            },
        };
        d.next_sid += 1;
        d.owners[c.slot] = o;
        errdefer d.owners[c.slot] = null; // a later error must not leave the freed owner reachable
        // (iii) THE FILL-LEAK CLOSE (the Flash Next errdefer form): an error past the admit ENDS the
        // stream — the Multi's queues must not hold a half-filled stream the release never reaches.
        errdefer d.mu.finish(sid);
        std.log.info("adapter: prefill-admitted sid={d} slot={d}", .{ sid, c.slot });
        // the prompt's chunks: THE MULTI'S OWN leadFill, one chunk a call over the Link (the op on
        // the wire before each collective chunk — the proven order, delegated not re-derived); the
        // loop bound is the filling queue itself, so exactly one op a chunk reaches the followers
        // and none past the end (a trailing op would find their queue empty and error OutOfStep).
        while (d.mu.filling.items.len > 0 and d.mu.filling.items[0].sid == sid)
            d.mu.leadFill(.{ .ctx = d, .fill = linkFill, .round = linkRound }) catch |e| {
                std.log.err("adapter: prefill fill failed sid={d}: {s}", .{ sid, @errorName(e) });
                return e;
            };
        // THE BOOTSTRAP: the first round's window must carry the drafter's count.
        try d.holdCount(o);
        std.log.info("adapter: prefill-bootstrapped", .{});
    }

    fn aVerify(p: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
        std.log.info("adapter: verify-entry", .{});
        const d: *Adapter = @ptrCast(@alignCast(p));
        d.wire.lockUncancelable(d.io);
        defer d.wire.unlock(d.io);
        const mu = d.mu;
        const g0 = mu.r.graph_runs; // §128.27en the split: this round's replay-or-miss delta
        const e0 = mu.r.eager_runs;
        var plan: multi.Plan = .{};
        var gmasks: [multi.max_streams]?[]u32 = @splat(null);
        defer {
            for (gmasks) |m| if (m) |x| d.gpa.free(x);
        }
        var gcons: [multi.max_streams]?*lanes.grammar.Constraint = @splat(null);
        var n_live: usize = 0;
        for (mu.streams.items) |x| n_live += @intFromBool(!x.done);
        for (windows, 0..) |w, i| {
            const o = d.ownerOfLane(w.stream) orelse return error.OutOfStep;
            const ms = mu.find(o.sid) orelse return error.OutOfStep;
            const arm: u32 = switch (ms.mode) {
                .mtp => multi.arm_mtp,
                .dflash, .dspark => multi.arm_draft,
                .serial => multi.arm_none,
            };
            const held: usize = @as(usize, w.held) + w.tokens.len;
            // THE ASK CLAMPED TO THE DRAFTER'S LIVE BUDGET: the count-hold sampled the live count
            // at draft time; an admission since shrank the mdraft's row budget (frows), and an ask
            // past it overflows the drafter's buffers — the illegal address the instant the second
            // stream joined. The delivery may come in under the ask still (the confidence floor
            // cuts); the map pads the window past it.
            const cap: usize = if (multi.drafterOf(ms.mode)) |which| mu.depthOf(which, n_live) else 0;
            const room: usize = if (arm == multi.arm_none) 0 else @min(held, cap);
            plan.items[plan.n] = .{ .sid = o.sid, .slot = @intCast(ms.slot), .P = @intCast(ms.P), .tok = ms.tok, .m = @intCast(ms.m), .mode = @backingInt(ms.mode), .arm = arm, .room = @intCast(room), .sampled = @intFromBool(ms.sampling != null) };
            if (o.lane.grammar) |g| {
                if (try g.next(d.gpa)) |m| {
                    plan.items[plan.n].grammar = @intCast(m.len);
                    plan.masks[plan.n] = m;
                    gmasks[plan.n] = m;
                    gcons[plan.n] = g;
                }
                std.log.info("adapter: mask-compute sid={d} grammar={d} active={}", .{ o.sid, plan.items[plan.n].grammar, g.active });
            }
            plan.n += 1;
            _ = i;
        }
        // THE VALIDATION (applyRound's head, verbatim): the plan is the truth the ranks agreed on;
        // a divergence from the stream's own state is the out-of-step class, loud — and it fires
        // BEFORE the wire op, so an error never deserts the follower mid-round.
        errdefer mu.broken = mu.world > 1;
        var ss: [multi.max_streams]*multi.Stream = undefined;
        for (plan.items[0..plan.n], 0..) |it, i| {
            const s = mu.find(it.sid) orelse return error.OutOfStep;
            if (!s.started or s.slot != it.slot or s.P != it.P or s.tok != it.tok or s.m != it.m or @backingInt(s.mode) != it.mode) {
                std.log.err("adapter: stream {d} is at ({d}, {d}, {d}), the round has ({d}, {d}, {d}): out of step", .{ s.sid, s.P, s.tok, s.m, it.P, it.tok, it.m });
                return error.OutOfStep;
            }
            ss[i] = s;
        }
        try linkRound(d, &plan);
        for (plan.items[0..plan.n]) |it|
            std.log.info("adapter: round-item sid={d} slot={d} P={d} tok={d} m={d} mode={d} arm={d} room={d} sampled={d}", .{ it.sid, it.slot, it.P, it.tok, it.m, it.mode, it.arm, it.room, it.sampled });
        if (mu.r.f.prof) |fp| fp.on = true; // 128.27fy the forward-prof armed for the decode windows (each round: the fill paths turn it off after their chunks)
        const pt0 = if (mu.pf.on) try mu.profBegin(&plan) else 0; // 128.27fs THE PROF WIRED ON THE ADAPTER S ROUND (the multi s own template: begin before gpu, the phases 0-3 inside gpu, mark(4) at picks end, mark(5)+end at commit)
        const t = try mu.gpu(&plan);
        {
            var vb_: [8]i64 = @splat(-1);
            const vn = @min(t.R, 8);
            @memcpy(vb_[0..vn], t.vids[0..vn]);
            var vbp: [8]i32 = @splat(-1);
            @memcpy(vbp[0..vn], t.vbase[0..vn]);
            const pst = &d.mu.r.st;
            const sstart = pst.slot_start;
            const srows = pst.slot_rows;
            std.log.info("adapter: round-tables R={d} M={d} S={d} ds0={d} off0={d} vids={any} vbase={any} local={d} sstart={any} srows={any}", .{ t.R, t.M, t.S, t.ds[0], t.offs[0], vb_[0..vn], vbp[0..vn], pst.local, if (sstart) |x| x[0..@min(x.len, 4)] else @as([]const usize, &.{}), if (srows) |x| x[0..@min(x.len, 4)] else @as([]const usize, &.{}) });
        }
        std.log.info("adapter: pre-picks n={d}", .{plan.n}); // §128.27hz the wedge hunt's stage marks
        try mu.picks(&plan, &t, &d.res_buf, &d.rows_buf);
        std.log.info("adapter: post-picks", .{});
        try mu.maskPicks(&plan, &t, &d.res_buf, &d.rows_buf);
        std.log.info("adapter: post-maskPicks", .{}); // §128.27hz the wedge hunt's stage marks
        for (plan.items[0..plan.n], 0..) |it, i| {
            if (it.grammar == 0) continue;
            const g = gcons[i] orelse return error.OutOfStep;
            if (!try g.preTake(d.res_buf[i].emit[0])) return error.WalkRefusedMaskedPick;
        }
        std.log.info("adapter: post-pretake", .{}); // §128.27hz the wedge hunt's stage marks
        // THE DELIVERED IS THE TRUTH: the drafter may deliver fewer drafts than the window holds —
        // the ask is the window's held count, the delivery the drafter's own (its confidence floor
        // cuts mid-run; its row budget caps by the live count). The rows past the delivery pad with
        // a token no vocabulary ever matches, so the Engine's acceptance path cuts exactly where the
        // drafter's confidence did: the committed picks and the Multi's own commit agree, the state
        // and the emission stay one. (The guard this replaces errored on the mismatch — and its
        // error path returned BEFORE the commit while the follower's .round handler committed
        // unconditionally: the wire diverged, the next collectives crossed into illegal addresses.)
        for (windows, out, 0..) |w, *o, i| {
            const rows = w.rows();
            const ds = t.ds[i];
            const off = t.offs[i];
            const pad: u32 = 0xffffffff; // never a token: a padded row's acceptance never holds
            for (o.sampled[0..rows], 0..) |*x, j| x.* = if (j <= ds) d.rows_buf[i][j] else pad;
            for (o.drafts[0 .. rows - 1], 0..) |*dr, j| dr.* = if (j < ds) @intCast(t.vids[off + 1 + j]) else pad;
        }
        // keep() is the no-op: this commit IS the keep (the no-keep wire beneath — the picks a
        // collective, every rank its own commit).
        try mu.mark(4);
        try mu.commit(&plan, &t, &d.res_buf);
        std.log.info("adapter: post-commit", .{}); // §128.27hz the wedge hunt's stage marks
        try mu.mark(5);
        if (mu.pf.on) mu.profEnd(pt0, t.R);
        // THE STATE ADVANCE (applyRound's tail, verbatim): the rounds before this landed the commits
        // but never moved the stream — P frozen at the prompt's end, every round re-verifying the
        // same window (the degenerate emission, the same picks forever), the repeated taps
        // overflowing the drafter's context buffer ten rounds in (the illegal address).
        mu.rounds += 1;
        for (plan.items[0..plan.n], 0..) |it, i| {
            const s = ss[i];
            const x = &d.res_buf[i];
            s.rounds += 1;
            s.drafted += t.ds[i];
            s.graph_runs += mu.r.graph_runs - g0;
            s.eager_runs += mu.r.eager_runs - e0;
            s.chain[@min(x.n, 3)] += 1;
            s.min_rows = if (s.min_rows == 0) 1 + t.ds[i] else @min(s.min_rows, 1 + t.ds[i]);
            s.P += x.n + 1;
            s.tok = x.emit[x.len - 1];
            s.m = x.n + 1;
            d.ds_buf[i] = t.ds[i];
            if (mu.dlog) |dl| dl.round(it.sid, it.slot, x.emit[0..x.len]);
        }
        std.log.info("adapter: round-done n={d}", .{plan.n});
    }

    /// keep(): the deliberate no-op — the commit IS the keep (the no-keep wire). But a TREE or a
    /// masked window is a form the no-keep never covered: refused LOUDLY, never a silent no-op.
    fn aKeep(_: *anyopaque, windows: []const lanes.backend.Window, _: []const []const u32) anyerror!void {
        for (windows) |w| {
            if (w.parents != null or w.masks.len > 0) {
                std.log.err("adapter: keep on a tree/masked window: the no-keep wire covers chains only", .{});
                return error.Unsupported;
            }
        }
    }

    fn aDraft(p: *anyopaque, requests: []const lanes.backend.DraftRequest) anyerror!void {
        std.log.info("adapter: draft-entry", .{});
        const d: *Adapter = @ptrCast(@alignCast(p));
        d.wire.lockUncancelable(d.io);
        defer d.wire.unlock(d.io);
        // the count-hold alone: the gpu's fused propose owns the drafting (the twins), no
        // collective ever outside the .round op.
        for (requests) |r| {
            const o = d.ownerOfLane(r.stream) orelse return error.OutOfStep;
            try d.holdCount(o);
        }
    }

    fn aRelease(p: *anyopaque, s: *lanes.Stream) void {
        std.log.info("adapter: release-entry", .{});
        const d: *Adapter = @ptrCast(@alignCast(p));
        d.wire.lockUncancelable(d.io);
        defer d.wire.unlock(d.io);
        const o = d.ownerOfLane(s) orelse return;
        if (d.mu.find(o.sid)) |ms| std.log.info("adapter: stream-done sid={d} rounds={d} g={d} e={d} c={d}/{d}/{d}/{d}", .{ o.sid, ms.rounds, ms.graph_runs, ms.eager_runs, ms.chain[0], ms.chain[1], ms.chain[2], ms.chain[3] });
        d.sendRaw(.finish, o.sid, 0) catch {};
        d.mu.finish(o.sid); // the pool's release
        d.freeOwner(o);
    }

    /// The first token: the fill's last chunk already drew it (applyFill sets the pending, s.tok) —
    /// the handle is the sid, read() returns the pending. No GPU work: the fill did it.
    fn aFirst(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const d: *Adapter = @ptrCast(@alignCast(p));
        d.wire.lockUncancelable(d.io);
        defer d.wire.unlock(d.io);
        const o = d.ownerOfLane(s) orelse return error.OutOfStep;
        const ms = d.mu.find(o.sid) orelse return error.OutOfStep;
        if (position != @as(u64, @intCast(ms.P))) {
            std.log.err("adapter: first draws at {d} but stream {d} sits at P={d}: out of step", .{ position, o.sid, ms.P });
            return error.OutOfStep;
        }
        return o.sid;
    }

    /// §128.27hz++++++++++++++++++++++++++++++ THE FIRST MASKED DRAW'S ADAPTER HALF: the fill
    /// ALREADY drew it — the admit's mask (the walk's root fill, the same bits the engine handed
    /// us: the same walk object, the same root state, next() pure) masked the fill's own pick on
    /// EVERY rank, the pick's gather a collective that a re-draw here would desert. The pending is
    /// the masked token; the handle returns, the engine reads it and commits it into the walk.
    fn aFirstMasked(p: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        _ = mask;
        return aFirst(p, s, position);
    }

    /// The serial one-token round: REFUSED — the Flash Next form (the refusal IS the contract,
    /// backend.zig:337); the hand-rolled one-row verify it replaces was re-derivation.
    fn aQueue(_: *anyopaque, _: *lanes.Stream, _: lanes.backend.Feed, _: u64) anyerror!u64 {
        return error.Unsupported;
    }

    /// A drawn token's value: the Multi's pending (the fill's draw, or the round's last accepted).
    fn aRead(p: *anyopaque, handle: u64) anyerror!u32 {
        const d: *Adapter = @ptrCast(@alignCast(p));
        d.wire.lockUncancelable(d.io);
        defer d.wire.unlock(d.io);
        const ms = d.mu.find(@intCast(handle)) orelse return error.OutOfStep;
        return ms.tok;
    }
};

const adapter_vtable = lanes.backend.Backend.VTable{
    .prefill = Adapter.aPrefill,
    .first = Adapter.aFirst,
    .first_masked = Adapter.aFirstMasked,
    .queue = Adapter.aQueue,
    .read = Adapter.aRead,
    .verify = Adapter.aVerify,
    .keep = Adapter.aKeep,
    .draft = Adapter.aDraft,
    .release = Adapter.aRelease,
};

// -- the vtable: the lane core's shared rounds are the v2 seam (cuda_multi's windows); the lone
//    driver above never touches these. Every stub refuses by name; release holds nothing per-stream.

fn stubPrefill(_: *anyopaque, _: *lanes.Stream) anyerror!void {
    return error.Glm53SharedRoundsAreV2;
}
fn stubFirst(_: *anyopaque, _: *lanes.Stream, _: u64) anyerror!u64 {
    return error.Glm53SharedRoundsAreV2;
}
fn stubQueue(_: *anyopaque, _: *lanes.Stream, _: lanes.backend.Feed, _: u64) anyerror!u64 {
    return error.NotPipelined;
}
fn stubRead(_: *anyopaque, _: u64) anyerror!u32 {
    return error.Glm53SharedRoundsAreV2;
}
fn stubVerify(_: *anyopaque, _: []const lanes.backend.Window, _: []lanes.backend.Verified) anyerror!void {
    return error.Glm53SharedRoundsAreV2;
}
fn stubKeep(_: *anyopaque, _: []const lanes.backend.Window, _: []const []const u32) anyerror!void {
    return error.Glm53SharedRoundsAreV2;
}
fn stubDraft(_: *anyopaque, _: []const lanes.backend.DraftRequest) anyerror!void {
    return error.Glm53SharedRoundsAreV2;
}
fn stubRelease(_: *anyopaque, _: *lanes.Stream) void {}

const vtable = lanes.backend.Backend.VTable{
    .prefill = stubPrefill,
    .first = stubFirst,
    .queue = stubQueue,
    .read = stubRead,
    .verify = stubVerify,
    .keep = stubKeep,
    .draft = stubDraft,
    .release = stubRelease,
};

fn close(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    var hdr = Hdr{ .magic = 0, .n_prompt = 0, .max_tokens = 0, .draft = 0 };
    own.boot.rdv.broadcast(std.mem.asBytes(&hdr)) catch {}; // the followers' shutdown signal
    own.session.deinit();
    own.boot.deinit();
    own.gpa.destroy(own.session);
    if (own.adapter) |ad| {
        ad.fini();
        own.gpa.destroy(ad);
    }
    own.gpa.destroy(own);
}

/// The server's words for this family's refusals.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NoRank => "GLM-5.3 (glm_moe_dsa) runs at TP over ranks: set TF_GLM53_RANK, TF_GLM53_WORLD and TF_GLM53_MASTER on every rank",
        error.Glm53SharedRoundsAreV2 => "the glm53r2 backend serves one drafted stream at a time (the lone driver); shared rounds are the v2 seam — serve --parallel 1",
        else => null,
    };
}
