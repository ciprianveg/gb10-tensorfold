//! THE V2 SEAM (§128.27ae-ah): the lane core's Backend over cuda_multi — the native server's shared rounds.
//!
//! The lane core's Engine is the shared-rounds master (admissions, fills, rounds, endings, the cost policy);
//! this adapter maps its Backend vtable onto the Multi's machinery, so concurrent requests ride the same
//! mixed-round verify windows the §109.4 receipts proved bit-identical ("a row computes exactly what it
//! computes alone" — the position-keyed kernels over the per-row tables).
//!
//! The mapping (the record's contract, §128.27ah):
//!   prefill  mu.admit (the lane Stream's prompt/sampling/eos) -> the applyFill chunk loop on the slot view
//!            -> pickFirst -> mu.start; the first token cached per slot.
//!   first    the cached first token (the handle = the token value; our draws complete synchronously).
//!   read     the identity.
//!   queue    the one-token round: a 1-row window through the same tail, committed immediately.
//!   draft    the host-staged propose path (the mdraft asks over the kept-in-sync drafter contexts),
//!            published on the lane Stream's own `next: Held`.
//!   verify   the small gpu tail: the windows -> the Plan items + the dd -> tables() (THE SAME FN the Multi's
//!            own step uses — the parity by construction) -> upload -> mirrorBounds -> mu.verify -> mu.picks
//!            (the per-row draws before the chain cut — the lane core's conclude decides the acceptance).
//!   keep     the paths -> the res -> mu.commit (the cache rollback + the drafter taps) + the state sync
//!            mirroring applyRound's tail (P/tok/m), so the next round's items pass the out-of-step guard.
//!   release  mu.finish + the slot free.
//!
//! The scope cut (the k=0 product form): the drafter arms only (dspark/dflash); the MTP arms refuse by name
//! (their chain is device-internal to the Multi's gpu — a follow-up only if a k>0 product form ever exists).
//! Device-held drafts (Window.held > 0) refuse at this landing: our drafts are host-staged.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;
const fw = @import("cuda_forward.zig");
const rn = @import("cuda_runner.zig");
const multi = @import("cuda_multi.zig");
const mdraft = @import("cuda_mdraft.zig");
const smp = @import("sampling.zig");
const tri = @import("cuda_triton.zig");

/// THE RANK COORDINATION'S WIRE (§128.27ak, the Link form): the message types shared by rank 0's
/// adapter (the sender) and cuda_native's followerV2 (the receiver). The v1's zero-magic shutdown
/// header terminates the v2 loop (the magic mismatch returns).
pub const v2_magic: u32 = 0x56533230;

pub const V2Kind = enum(u32) { shutdown = 0, admit = 1, fill = 2, round = 3, finish = 4, keep = 5, heartbeat = 6 };

pub const V2Msg = extern struct {
    magic: u32,
    kind: u32,
    sid: u32,
    n: u32,
    pad: [10]u32 = @splat(0), // the §98 Hdr's size (56): ONE receive serves both wires (the unified follower)
};

/// the admit's mirrored fields (the sampling included — the follower's picks draw with it).
/// §128.27hz+++++++++++ THE RESUME ON THE WIRE: begin (the leader's authoritative resume point)
/// + the keeps (keep_most=3, the fixed slack 4) — the stops recompute deterministically on every
/// rank from the mirrored plan (plan_rz.points over the same prompt, filtered (begin, L)).
pub const V2Admit = extern struct {
    slot: u32,
    mode: u32,
    count: u32,
    has_sampling: u32,
    seed: u64,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    begin: u32 = 0,
    keeps_n: u32 = 0,
    keeps: [4]u32 = @splat(0),
    /// §128.27hz+++++++++++++++++++++ THE RESUME'S SOURCE ON THE WIRE: the leader's busy-else branch
    /// resumes INTO a fresh slot FROM the hit's busy one (c.src) — the followers' namedIn looks there.
    /// The field was missing: the followers fell back to their own slot and died ReuseOutOfStep at
    /// every grown turn whose source was still decoding (the 1195 receipt: the entry at slot 0, the
    /// lookup at slot 1). 0 = no source (the in-place resume); the slots are < 32 so 0 is unambiguous
    /// with has_src as the flag.
    has_src: u32 = 0,
    src: u32 = 0,
    /// §128.27hz++++++++++++++++++++++++++++++ THE FIRST TOKEN'S MASK ON THE WIRE: the grammar's
    /// root fill rides the admit (prompt-independent, computed once) — the followers' fill replays
    /// the SAME masked first pick, the pick's gather never deserted. has_grammar gates a follow-up
    /// broadcast of grammar_words u32s, right after this struct.
    has_grammar: u32 = 0,
    grammar_words: u32 = 0,
};

/// lanes.Sampling -> smp.Sampling (cuda_native's field-identical map).
pub fn toOur(s: ?lanes.Sampling) ?smp.Sampling {
    const x = s orelse return null;
    if (x.temperature <= 0) return null; // greedy, the anchors' path
    return .{ .seed = x.seed, .temperature = x.temperature, .top_k = @intCast(x.top_k), .top_p = x.top_p, .min_p = x.min_p };
}

pub const Adapter = struct {
    gpa: std.mem.Allocator,
    mu: *multi.Multi,
    rdv: ?*cuda.rendezvous.Rendezvous = null, // rank 0's broadcast (the Link); null in tests/CLI
    /// §128.27hz++++++++++++++++++++++++++++++ the round message's buffer: the plan's words plus
    /// the grammar pool (every stream structured at once — the realistic one or two a rounding
    /// error away)
    plan_words_buf: [multi.plan_words + multi.grammar_pool_words]u32 = @splat(0),
    owners: [multi.max_streams]?*Stream = @splat(null),
    sids: [multi.max_streams]u32 = @splat(0),
    first_tok: [multi.max_streams]?u32 = @splat(null),
    next_sid: u32 = 1,
    /// the last verify round's plan + tables, held for keep() (the commit needs both).
    plan: multi.Plan = .{},
    tables: ?multi.Tables = null,
    /// the last round's per-row draws (picks' out param), kept for keep()'s res assembly.
    drew: [multi.max_streams][fw.decode_rows]u32 = @splat(@splat(0)),
    /// the last round's host-staged drafts (the Verified.drafts the lane core reads are a mutable view).
    dd: multi.Drafts = .{},
    /// §128.27am THE ALWAYS-KEEP: the lane core's familyRound skips keep() on a full-length chain keep
    /// (engine.zig:209 - no rollback needed); the adapter must still commit, sync and broadcast, so each
    /// verify() arms the flag and the next round flushes the full keep if keep() never came.
    keep_sent: bool = true,

    fn self(ptr: *anyopaque) *Adapter {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn init(gpa: std.mem.Allocator, mu: *multi.Multi, rdv: ?*cuda.rendezvous.Rendezvous) Adapter {
        return .{ .gpa = gpa, .mu = mu, .rdv = rdv };
    }

    /// rank 0's Link send (a no-op without the rendezvous).
    fn sendRaw(b: *Adapter, kind: V2Kind, sid: u32, n: u32) !void {
        const rdv = b.rdv orelse return;
        var msg = V2Msg{ .magic = v2_magic, .kind = @intFromEnum(kind), .sid = sid, .n = n };
        try rdv.broadcast(std.mem.asBytes(&msg));
    }

    /// §128.27an THE WIRE-ORDER GATE: the pending keep flushes before ANY gated v2 message - the
    /// Engine's next action after a skipped keep can be a fill or a finish, not a verify, and the
    /// follower waits at its keep-receive (a foreign header there errors it out of the loop: the
    /// 5-token freeze). The keep's own header rides sendRaw - the cycle broken structurally.
    fn send(b: *Adapter, kind: V2Kind, sid: u32, n: u32) !void {
        if (!b.keep_sent and kind != .keep) try b.flushFullKeep();
        try b.sendRaw(kind, sid, n);
    }

    /// THE LINK's round message: the header then the plan's words (the codec the Multi already owns).
    fn sendPlan(b: *Adapter, plan: *const multi.Plan) !void {
        const rdv = b.rdv orelse return;
        const words = try plan.encode(&b.plan_words_buf);
        try b.send(.round, 0, @intCast(words.len));
        try rdv.broadcast(std.mem.sliceAsBytes(words));
    }

    /// THE LINK's keep message (§128.27al, the OutOfStep fix): the LEAD's acceptance - the lane core's
    /// conclude keeps by its own policy, not the Multi's chain cut, so the follower's commit and state
    /// sync must take the lead's res, not its own. The words: per item [n, len, emit[0..len]].
    fn sendKeep(b: *Adapter, plan: *const multi.Plan, res: *const [multi.max_streams]multi.Result) !void {
        const rdv = b.rdv orelse return;
        var words: [multi.max_streams * (2 + fw.decode_rows)]u32 = @splat(0);
        var at: usize = 0;
        for (0..plan.n) |i| {
            const x = &res[i];
            words[at] = @intCast(x.n);
            words[at + 1] = @intCast(x.len);
            @memcpy(words[at + 2 ..][0..x.len], x.emit[0..x.len]);
            at += 2 + x.len;
        }
        try b.sendRaw(.keep, 0, @intCast(at));
        try rdv.broadcast(std.mem.sliceAsBytes(words[0..at]));
    }

    pub fn backend(b: *Adapter) be.Backend {
        return .{ .ptr = b, .vtable = &.{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .release = releaseFn,
        } };
    }

    /// What the round loop reads (the reference's shape): the drafter arms, chains over the shared windows.
    /// The window costs stay empty at this landing (the depth rule's defaults); the drafts figure is the
    /// drafter's most — the adaptive share is the Multi's own depthOf at round time.
    pub fn facts(b: *const Adapter) lanes.Model {
        const dspark = b.mu.md[1] != null or b.mu.md[0] != null;
        return .{
            .exact_width = fw.decode_rows,
            .mtp = false,
            .speculate = dspark,
            .speculate_early = false,
            .drafts = if (dspark) 7 else 0,
            .mtp_step_ms = 0,
            .draft_streams = dspark,
            .window_costs = &.{},
            .hidden_rows = true,
            .streams_exact = true,
            .batch_rows = fw.decode_rows,
            .max_streams = b.mu.N,
            .lone_copies = false,
        };
    }

    fn slotOf(b: *const Adapter, s: *const Stream) ?usize {
        for (&b.owners, 0..) |o, i| if (o == s) return i;
        return null;
    }

    fn freeSlot(b: *const Adapter) ?usize {
        for (0..b.mu.N) |i| {
            if (b.owners[i] != null or b.mu.busy[i]) continue;
            return i;
        }
        return null;
    }

    /// The small gpu tail minus propose: tables -> upload -> the mirror bounds -> the shared window forward.
    /// The plan's items must already mirror the Multi streams' state (the out-of-step guard's own check).
    fn forward(b: *Adapter, plan: *const multi.Plan, dd: *const multi.Drafts) !multi.Tables {
        var t = try multi.tables(plan, b.mu.k, b.mu.BL, b.mu.local, dd);
        if (t.R == 0) return error.Invalid;
        try b.mu.upload(&t);
        try b.mu.mirrorBounds(plan, &t);
        var sampled = false;
        for (plan.items[0..plan.n]) |it| sampled = sampled or it.sampled != 0;
        const T = tri.bucket(t.end, b.mu.r.topk);
        try b.mu.verify(t.R, T, if (sampled) .local else .argmax);
        return t;
    }

    fn itemOf(b: *Adapter, slot: usize, w: *const be.Window, drafts: usize) !multi.Item {
        const ms = b.mu.find(b.sids[slot]) orelse return error.UnknownStream;
        if (ms.mode == .mtp) return error.Glm53MtpArmsAreV2FollowUp; // the k=0 scope cut
        const P: u64 = w.positions[0];
        // §128.27am THE SYNC-FROM-WINDOW: the lane core is the authority - its positions/pending are the
        // truth (the familyRound's skipped keeps made the assert class fail); the adapter mirrors.
        ms.P = @intCast(P);
        ms.tok = w.pending;
        const arm: u32 = if (ms.mode == .serial) multi.arm_none else multi.arm_draft;
        return .{ .sid = ms.sid, .slot = @intCast(slot), .P = @intCast(P), .tok = w.pending, .m = @intCast(ms.m), .mode = @backingInt(ms.mode), .arm = arm, .room = @intCast(drafts), .sampled = @intFromBool(w.stream.sampling != null) };
    }

    // -- prefill: admit -> the chunk loop on the slot view -> pickFirst -> start

    fn prefillFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const b = self(ptr);
        const slot = b.freeSlot() orelse return error.NoFreeSlot;
        const ids = s.prompt();
        if (ids.len == 0) return error.PromptTooLong;
        const sid = b.next_sid;
        const a_mode: multi.Mode = if (s.drafts) .dspark else .serial;
        const a_count: usize = @as(usize, s.max_new) + 2; // the lane core owns the ending; the Multi's count is inert here
        // THE LINK (§128.27ak): the admit first (the header, the prompt, the fields), then the local admit
        try b.send(.admit, sid, @intCast(ids.len));
        if (b.rdv) |rdv| {
            const wire = try b.gpa.dupe(u32, ids); // the broadcast takes a mutable slice
            defer b.gpa.free(wire);
            try rdv.broadcast(std.mem.sliceAsBytes(wire));
            const sm = toOur(s.sampling);
            var aw = V2Admit{
                .slot = @intCast(slot),
                .mode = @intFromEnum(a_mode),
                .count = @intCast(a_count),
                .has_sampling = @intFromBool(sm != null),
                .seed = if (sm) |x| x.seed else 0,
                .temperature = if (sm) |x| @floatCast(x.temperature) else 0,
                .top_k = if (sm) |x| @intCast(x.top_k) else 0,
                .top_p = if (sm) |x| @floatCast(x.top_p) else 0,
                .min_p = if (sm) |x| @floatCast(x.min_p) else 0,
            };
            try rdv.broadcast(std.mem.asBytes(&aw));
        }
        const ms = try b.mu.admit(.{
            .sid = sid,
            .slot = slot,
            .mode = a_mode,
            .sampling = toOur(s.sampling),
            .count = a_count,
            .prompt = ids,
            .eos = s.eos,
        });
        b.next_sid += 1;
        b.owners[slot] = s;
        b.sids[slot] = ms.sid;
        errdefer {
            b.owners[slot] = null;
            b.mu.finish(ms.sid);
        }
        // the chunks a lone request takes, back to back, until the first token (applyFill's own path);
        // THE LINK: each chunk broadcast before it runs (the sequence pinned, the collectives meet)
        var tok: ?u32 = null;
        while (tok == null) {
            try b.send(.fill, ms.sid, 0);
            tok = try b.mu.applyFill(ms.sid);
        }
        b.first_tok[slot] = tok;
    }

    fn firstFn(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        const tok = b.first_tok[slot] orelse return error.NoFirstToken;
        _ = position; // the fill already drew at the prompt's end; the position is the adapter's own state
        return tok;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        _ = ptr;
        return @intCast(handle); // our draws complete synchronously: the handle is the token
    }

    /// The one-token round (the serial streams' path): a 1-row window through the same tail, committed now.
    fn queueFn(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        const ms = b.mu.find(b.sids[slot]) orelse return error.UnknownStream;
        const fed: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        if (fed != ms.tok) return error.OutOfStep; // the fed token is the pending we hold
        _ = position;
        var plan: multi.Plan = .{};
        plan.items[0] = .{ .sid = ms.sid, .slot = @intCast(slot), .P = @intCast(ms.P), .tok = ms.tok, .m = @intCast(ms.m), .mode = @backingInt(ms.mode), .arm = multi.arm_none, .room = 0, .sampled = @intFromBool(s.sampling != null) };
        plan.n = 1;
        try b.sendPlan(&plan);
        var dd: multi.Drafts = .{};
        var t = try b.forward(&plan, &dd);
        var res: [multi.max_streams]multi.Result = undefined;
        var drew: [multi.max_streams][fw.decode_rows]u32 = @splat(@splat(0));
        try b.mu.picks(&plan, &t, &res, &drew);
        res[0] = .{ .n = 0, .len = 1 };
        res[0].emit[0] = drew[0][0];
        try b.mu.commit(&plan, &t, &res);
        ms.P += 1;
        ms.tok = drew[0][0];
        ms.m = 1;
        try b.sendKeep(&plan, &res); // THE LINK: the one-token round's keep
        b.keep_sent = true;
        return drew[0][0];
    }

    // -- verify: the windows -> the plan + dd -> the tail -> the per-row draws

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        if (windows.len == 0 or windows.len > multi.max_streams or windows.len != out.len) return error.BadWindows;
        try b.flushFullKeep(); // §128.27am: the lane core may have skipped the last round's keep
        var plan: multi.Plan = .{};
        b.dd = .{};
        for (windows, 0..) |w, i| {
            if (w.held > 0) return error.HeldDrafts; // this landing: host-staged drafts only
            const slot = b.slotOf(w.stream) orelse return error.UnknownStream;
            const d = w.tokens.len;
            if (d > fw.decode_rows - 1) return error.WindowTooWide;
            plan.items[i] = try b.itemOf(slot, &w, d);
            b.dd.n[i] = d;
            @memcpy(b.dd.ids[i][0..d], w.tokens);
        }
        plan.n = windows.len;
        try b.sendPlan(&plan);
        var t = try b.forward(&plan, &b.dd);
        var res: [multi.max_streams]multi.Result = undefined;
        try b.mu.picks(&plan, &t, &res, &b.drew);
        for (windows, 0..) |_, i| {
            const d = t.ds[i];
            if (out[i].sampled.len < 1 + d) return error.BadWindows;
            @memcpy(out[i].sampled[0 .. 1 + d], b.drew[i][0 .. 1 + d]);
            out[i].drafts = b.dd.ids[i][0..d]; // the drafts the forward verified (the adapter's mutable view)
        }
        b.plan = plan;
        b.tables = t;
        b.keep_sent = false; // armed: the next round flushes the full keep if keep() never comes
    }

    // -- keep: the paths -> the res -> the commit + the state sync (applyRound's tail)

    /// One keep applied: the commit, the per-slot state sync (applyRound's tail), the Link broadcast.
    fn applyKeep(b: *Adapter, res: *const [multi.max_streams]multi.Result) !void {
        try b.mu.commit(&b.plan, &(b.tables orelse return error.NoRound), res);
        for (b.plan.items[0..b.plan.n], 0..) |it, i| {
            const ms = b.mu.find(it.sid) orelse return error.UnknownStream;
            const x = &res[i];
            ms.P += x.n + 1;
            ms.tok = x.emit[x.len - 1];
            ms.m = x.n + 1;
        }
        b.keep_sent = true; // raised BEFORE the sendKeep: the send gate's recursion guard
        try b.sendKeep(&b.plan, res);
    }

    /// §128.27am: the full keep the lane core skipped (the familyRound's no-rollback contract) - all the
    /// round's rows kept: n = the drafts, len = 1 + the drafts, emit = the recorded draws.
    fn flushFullKeep(b: *Adapter) !void {
        if (b.keep_sent) return;
        const t = &(b.tables orelse {
            b.keep_sent = true;
            return;
        });
        var res: [multi.max_streams]multi.Result = @splat(.{});
        for (0..b.plan.n) |i| {
            const d = t.ds[i];
            res[i] = .{ .n = d, .len = 1 + d };
            @memcpy(res[i].emit[0 .. 1 + d], b.drew[i][0 .. 1 + d]);
        }
        try b.applyKeep(&res);
        b.tables = null;
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        if (windows.len != paths.len) return error.BadPaths;
        _ = b.tables orelse return error.NoRound;
        var res: [multi.max_streams]multi.Result = undefined;
        for (windows, 0..) |w, i| {
            _ = w;
            const kept = paths[i].len; // the kept rows: a chain's prefix count as the path's length
            if (kept == 0 or kept > fw.decode_rows) return error.BadPaths;
            res[i] = .{ .n = kept - 1, .len = kept };
            @memcpy(res[i].emit[0..kept], b.drew[i][0..kept]);
        }
        try b.applyKeep(&res);
        b.tables = null;
    }

    // -- draft: the host-staged propose path, published on the lane Stream's next: Held

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        for (requests) |q| {
            const slot = b.slotOf(q.stream) orelse return error.UnknownStream;
            const ms = b.mu.find(b.sids[slot]) orelse return error.UnknownStream;
            if (ms.mode == .mtp) return error.Glm53MtpArmsAreV2FollowUp;
            // free the previous Held's tokens, then publish the new (the lane core reads `next` for its window)
            if (q.stream.next) |old| {
                if (old.tokens) |ot| b.gpa.free(ot);
                q.stream.next = null;
            }
            if (ms.mode == .serial) continue; // nothing to draft
            const which = multi.drafterOf(ms.mode) orelse continue;
            const x = b.mu.md[which] orelse continue;
            const n_live = b.mu.live();
            // the Multi's own room form: the adaptive share, the tokens left, the cache left
            const depth = @min(@min(b.mu.depthOf(which, n_live), ms.count -| ms.out.items.len -| 1), b.mu.r.capacity -| ms.P -| 1);
            const want = @min(@as(usize, q.depth), depth);
            if (q.depth == 0 or want == 0) continue;
            const asks = [_]mdraft.Ask{.{ .slot = slot, .pending = ms.tok, .depth = want, .sampling = ms.sampling, .conf = b.mu.confOf(which, 0) }};
            var got: [1]mdraft.Drafted = undefined;
            try x.propose(&asks, &got);
            if (got[0].n == 0) continue;
            const toks = try b.gpa.dupe(u32, got[0].slice());
            q.stream.next = .{ .count = @intCast(got[0].n), .tokens = toks };
        }
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return;
        b.flushFullKeep() catch {}; // the follower may wait at the keep receive; resolve it first
        if (s.next) |old| {
            if (old.tokens) |ot| b.gpa.free(ot);
            s.next = null;
        }
        b.send(.finish, b.sids[slot], 0) catch {}; // the void vtable slot: a failed send rides the shutdown
        b.mu.finish(b.sids[slot]);
        b.owners[slot] = null;
        b.first_tok[slot] = null;
    }
};
