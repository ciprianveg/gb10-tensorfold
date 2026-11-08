//! §98 THE HTTP SURFACE (the minimal Phase-3b serve): rank 0's HTTP side over the served `Boot` + `Session`
//! — the engine's own design ("rank 0's HTTP side, ranks 1..N following its requests") — with the §98 binder
//! load-branch (TF_GLM53_BIND) as the weights source. One request at a time, greedy; requests distribute
//! through the Boot's rendezvous broadcast and every rank runs `Session.exec` alike. The full native-server
//! registry integration (server/engines.zig -> native.open -> the lanes backend) stays the named product
//! item; this is the bench lane's serve surface: enough for the served-boot coherence gates, the serve-mode
//! SP numbers, and the HTTP form of the A4a-class checks.
//!
//! `tf-glm53r2 serve <model-dir> <port> [context]` — the settings from TF_GLM53_RANK/_WORLD/_MASTER/...
//! exactly as Options.fromEnv reads them (the script's names).

const std = @import("std");
const posix = std.posix;
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const r2 = @import("glm53r2");

const max_head = 64 * 1024;
const max_body = 8 * 1024 * 1024;

pub fn serve(gpu: check.Gpu, args: []const []const u8) !void {
    const a = gpu.gpa;
    if (args.len < 2) return error.Usage;
    const dir = args[0];
    const port = try std.fmt.parseInt(u16, args[1], 10);
    const context: usize = if (args.len > 2) std.fmt.parseInt(usize, args[2], 10) catch 32768 else 32768;

    const t0 = check.now(gpu.io);
    const o = try r2.engine.Options.fromEnv(dir, context);
    const boot = try r2.engine.Boot.init(a, gpu.io, o);
    defer boot.deinit();
    var s = try r2.engine.Session.init(a, boot);
    defer s.deinit();
    std.debug.print("SERVE boot: rank {d}/{d} up in {d} s (context {d}, k {d}, kv8 {}, dcp {d}, sp {}, graphs {}, load {d:.0} s)\n", .{ o.rank, o.world, @divTrunc(check.now(gpu.io) - t0, 1_000_000_000), o.context, o.k, o.kv8, boot.dcp, o.prompt_sp, o.graphs, boot.load_s });

    // the tokenizer (the pack carries none; the served checkpoint beside it does — the Runner's resolution)
    var tok_buf: [1024]u8 = undefined;
    const tok_dir: []const u8 = if (std.c.getenv("TF_GLM53_TOKENIZER")) |p| std.mem.span(p) else try std.fmt.bufPrint(&tok_buf, "{s}/../glm53-nvfp4-vision", .{dir});
    var tok = try core.tokenizer.loadTokenizer(gpu.io, a, tok_dir);
    defer tok.deinit();

    if (o.rank == 0) {
        try rank0(a, gpu.io, boot, &s, &tok, port);
    } else {
        try follower(&s);
    }
    std.debug.print("SERVE rank {d}: down\n", .{o.rank});
}

/// The request header every rank executes: the prompt's length, the reply budget, the draft mode.
/// §125t4: the reuse choice (rank 0's PromptReuse.choose — the engine's design: rank 0 picks the resume
/// point and the keep points and shares them with the request) and the sampled arm's temperature in
/// milli-units (0: greedy — the anchors' lane) ride it; the stops/keeps arrays follow the ids broadcast.
const Hdr = extern struct { magic: u32, n_prompt: u32, max_tokens: u32, draft: u32, begin: u32, n_stops: u32, n_keeps: u32, temperature_milli: u32 };
const hdr_magic: u32 = 0x51455231; // "1REQ"

/// Ranks 1..: apply rank 0's requests, forever (the container's rm is the shutdown).
fn follower(s: *r2.engine.Session) !void {
    const a = s.gpa;
    while (true) {
        var hdr: Hdr = undefined;
        try s.boot.rdv.broadcast(std.mem.asBytes(&hdr));
        if (hdr.magic != hdr_magic) return error.BadBroadcast;
        const ids = try a.alloc(u32, hdr.n_prompt);
        defer a.free(ids);
        try s.boot.rdv.broadcast(std.mem.sliceAsBytes(ids));
        const stops = try a.alloc(usize, hdr.n_stops);
        defer a.free(stops);
        try s.boot.rdv.broadcast(std.mem.sliceAsBytes(stops));
        const keeps = try a.alloc(u32, hdr.n_keeps);
        defer a.free(keeps);
        try s.boot.rdv.broadcast(std.mem.sliceAsBytes(keeps));
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(a);
        _ = try s.exec(.{ .prompt = ids, .max_tokens = hdr.max_tokens, .draft = hdr.draft != 0, .eos = s.eos, .copies = s.boot.o.copy.on, .begin = hdr.begin, .stops = stops, .keeps = keeps, .sampling = samplingOf(ids, hdr.temperature_milli) }, &out);
        if (out.items.len == 0) std.debug.print("SERVE rank {d}: empty reply\n", .{s.boot.o.rank});
    }
}

fn rank0(a: std.mem.Allocator, io: std.Io, boot: *r2.engine.Boot, s: *r2.engine.Session, tok: anytype, port: u16) !void {
    _ = boot;
    const sock = try listen(port);
    defer _ = std.c.close(sock);
    std.debug.print("SERVE up: http://0.0.0.0:{d}/generate  (POST {{\"prompt\": \"...\", \"max_tokens\": N, \"draft\": true}}; draft=false is the k=0 serial arm)\n", .{port});
    var served: usize = 0;
    while (true) {
        const rc = posix.system.accept(sock, null, null);
        if (posix.errno(rc) != .SUCCESS) {
            std.debug.print("SERVE accept: errno {d}\n", .{posix.errno(rc)});
            std.Io.sleep(io, .fromMilliseconds(200), .awake) catch {};
            continue;
        }
        const fd: posix.socket_t = @intCast(rc);
        handle(a, io, s, tok, fd) catch |e| std.debug.print("SERVE req {d}: {t}\n", .{ served, e });
        _ = std.c.close(fd);
        served += 1;
    }
}

fn listen(port: u16) !posix.socket_t {
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const sock: posix.socket_t = @intCast(rc);
    errdefer _ = std.c.close(sock);
    posix.setsockopt(sock, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&@as(c_int, 1))) catch {};
    const sa: posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = 0 }; // INADDR_ANY
    if (posix.errno(posix.system.bind(sock, @ptrCast(&sa), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.BindFailed;
    if (posix.errno(posix.system.listen(sock, 16)) != .SUCCESS) return error.ListenFailed;
    return sock;
}

fn sysRead(fd: posix.socket_t, buf: []u8) !usize {
    const rc = posix.system.read(fd, buf.ptr, buf.len);
    switch (posix.errno(rc)) {
        .SUCCESS => return @intCast(rc),
        .INTR => return 0,
        else => return error.Closed,
    }
}

fn writeAll(fd: posix.socket_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const rc = posix.system.write(fd, bytes[done..].ptr, bytes.len - done);
        switch (posix.errno(rc)) {
            .SUCCESS => done += @intCast(rc),
            .INTR => {},
            else => return error.Closed,
        }
    }
}

fn respond(fd: posix.socket_t, code: u16, msg: []const u8) !void {
    var buf: [512]u8 = undefined;
    const head = try std.fmt.bufPrint(&buf, "HTTP/1.1 {d} \r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{ code, msg.len });
    try writeAll(fd, head);
    try writeAll(fd, msg);
}

const Req = struct {
    prompt: []const u8,
    max_tokens: usize = 48,
    draft: bool = true,
    /// §125t4: the sampled arm (absent or <= 0: greedy — the anchors' lane). The §123 card class:
    /// the seeded per-position draws (seedFor(prompt_ids, TF_GLM53_SEED_SALT)), no top_k, top_p 1.0.
    temperature: f64 = 0,
};

/// §125t4: the request's sampling from the header's temperature (milli-units; 0: greedy). The seed is
/// keyed by the prompt ids + the boot's salt — the same on every rank, so the draws are deterministic
/// and a resumed request's fresh tail lands on the same keyed positions as a fresh run's.
fn samplingOf(ids: []const u32, temperature_milli: u32) ?r2.sampling.Sampling {
    if (temperature_milli == 0) return null;
    var salt: i64 = 0;
    if (std.c.getenv("TF_GLM53_SEED_SALT")) |v| salt = std.fmt.parseInt(i64, std.mem.span(v), 10) catch 0;
    return .{ .seed = r2.sampling.seedFor(ids, salt), .temperature = @as(f64, @floatFromInt(temperature_milli)) / 1000.0, .top_k = 0, .top_p = 1.0 };
}

fn handle(a: std.mem.Allocator, io: std.Io, s: *r2.engine.Session, tok: anytype, fd: posix.socket_t) !void {
    var head: [max_head]u8 = undefined;
    var got: usize = 0;
    var head_end: usize = 0;
    while (true) {
        const n = sysRead(fd, head[got..]) catch return error.Closed;
        if (n == 0) return error.Closed;
        got += n;
        if (std.mem.indexOf(u8, head[0..got], "\r\n\r\n")) |i| {
            head_end = i + 4;
            break;
        }
        if (got == head.len) return respond(fd, 431, "{\"error\": \"head too big\"}");
    }
    if (!std.mem.startsWith(u8, head[0..head_end], "POST ")) return respond(fd, 405, "{\"error\": \"POST only\"}");
    const path_ok = std.mem.startsWith(u8, head[5..head_end], "/generate");
    if (!path_ok) return respond(fd, 404, "{\"error\": \"POST /generate\"}");
    const clen = contentLength(head[0..head_end]) orelse return respond(fd, 411, "{\"error\": \"no Content-Length\"}");
    if (clen > max_body) return respond(fd, 413, "{\"error\": \"body too big\"}");
    const body = try a.alloc(u8, clen);
    defer a.free(body);
    const have = @min(got - head_end, clen);
    @memcpy(body[0..have], head[head_end .. head_end + have]);
    var done = have;
    while (done < clen) {
        const n = sysRead(fd, body[done..]) catch return error.Closed;
        if (n == 0) return error.Closed;
        done += n;
    }

    const req = std.json.parseFromSliceLeaky(Req, a, body, .{}) catch return respond(fd, 400, "{\"error\": \"bad json\"}");
    const ids = try tok.encode(a, req.prompt);
    defer a.free(ids);
    if (ids.len == 0 or ids.len >= s.boot.o.context) return respond(fd, 400, "{\"error\": \"prompt length\"}");
    const max_tokens = s.clampTokens(ids.len, req.max_tokens) catch return respond(fd, 400, "{\"error\": \"prompt too long for the context\"}");

    // §125t4: rank 0 picks the reuse choice (the resume point + the cuts + the kept points) and shares
    // it with the request. The raw no-marker prompts of the anchor class take the empty choice — begin 0,
    // no cuts — the same path §124's P1 anchors ran (pr.run with the empty stops), bit for bit. At the
    // multi boot the store rides the multi instead: the empty choice, the plain run, the v2 seam's item.
    const ch = try s.choose(ids, req.draft);
    defer ch.deinit(a);
    const temperature_milli: u32 = if (req.temperature > 0) @intFromFloat(@min(req.temperature, 25.0) * 1000.0) else 0;

    // the request to every rank: the header, then the prompt ids, then the choice (the rendezvous holds them in step)
    const hdr = Hdr{ .magic = hdr_magic, .n_prompt = @intCast(ids.len), .max_tokens = @intCast(max_tokens), .draft = if (req.draft) 1 else 0, .begin = @intCast(ch.begin), .n_stops = @intCast(ch.stops.len), .n_keeps = @intCast(ch.keeps.len), .temperature_milli = temperature_milli };
    try s.boot.rdv.broadcast(std.mem.asBytes(@constCast(&hdr)));
    try s.boot.rdv.broadcast(std.mem.sliceAsBytes(@constCast(ids)));
    try s.boot.rdv.broadcast(std.mem.sliceAsBytes(@constCast(ch.stops)));
    try s.boot.rdv.broadcast(std.mem.sliceAsBytes(@constCast(ch.keeps)));

    const t0 = check.now(io);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    const fin = try s.exec(.{ .prompt = ids, .max_tokens = max_tokens, .draft = req.draft, .eos = s.eos, .copies = s.boot.o.copy.on, .begin = ch.begin, .stops = ch.stops, .keeps = ch.keeps, .sampling = samplingOf(ids, temperature_milli) }, &out); // §100: the copy drafts follow the boot's settings (the default-true here was the §98.6 crash: the 16-row MTP copy window hit the missing variant)
    const wall_s = @as(f64, @floatFromInt(check.now(io) - t0)) / 1e9;
    const st = fin.stats;

    const text = try tok.decode(a, out.items, true);
    defer a.free(text);
    const k = if (req.draft) s.boot.o.k else 0;
    // §106: the drafter chain (k=0 + TF_GLM53_DSPARK) reports the full depth histogram — the dspark block drafts up to 8, the [9] bucket set covers it
    const hw: usize = if (req.draft and s.boot.o.dspark.len > 0) st.depth_breaks.len else k + 1;
    const accept: f64 = if (st.drafted > 0) @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(st.drafted)) else 0;
    const tok_s: f64 = if (st.decode_s > 0) @as(f64, @floatFromInt(out.items.len)) / st.decode_s else 0;
    const show = text[0..@min(text.len, 120)];
    std.debug.print("SERVE req: prompt={d} cached={d} gen={d} k={d} prefill_s={d:.3} decode_s={d:.3} wall_s={d:.3} tok_s={d:.2} accept={d:.3} depth_breaks={any} out={s}{s}\n", .{ ids.len, ch.begin, out.items.len, k, st.prefill_s, st.decode_s, wall_s, tok_s, accept, st.depth_breaks[0..hw], show, if (text.len > show.len) " [truncated]" else "" });

    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    const w = &aw.writer;
    try w.print("{{\"cached\":{d},\"prompt_tokens\":{d},\"gen\":{d},\"k\":{d},\"prefill_s\":{d:.3},\"decode_s\":{d:.3},\"wall_s\":{d:.3},\"tok_s\":{d:.2},\"rounds\":{d},\"drafted\":{d},\"accepted\":{d},\"accept\":{d:.3},\"depth_breaks\":[", .{ ch.begin, ids.len, out.items.len, k, st.prefill_s, st.decode_s, wall_s, tok_s, st.rounds, st.drafted, st.accepted, accept });
    for (st.depth_breaks[0..hw], 0..) |c, i| try w.print("{s}{d}", .{ if (i > 0) "," else "", c });
    try w.writeAll("],\"tokens\":[");
    for (out.items, 0..) |t, i| try w.print("{s}{d}", .{ if (i > 0) "," else "", t });
    try w.writeAll("],\"text\":\"");
    try jsonEscape(w, text);
    try w.writeAll("\"}\n");

    var http: [256]u8 = undefined;
    const h = try std.fmt.bufPrint(&http, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{aw.written().len});
    try writeAll(fd, h);
    try writeAll(fd, aw.written());
}

fn jsonEscape(w: *std.Io.Writer, text: []const u8) !void {
    for (text) |c| {
        switch (c) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => {
                if (c < 0x20) {
                    try w.print("\\u{x:0>4}", .{c});
                } else {
                    try w.writeByte(c);
                }
            },
        }
    }
}

fn contentLength(head: []const u8) ?usize {
    var it = std.mem.splitSequence(u8, head, "\r\n");
    while (it.next()) |line| {
        if (line.len < 16) continue;
        if (!std.ascii.startsWithIgnoreCase(line, "Content-Length:")) continue;
        return std.fmt.parseInt(usize, std.mem.trim(u8, line["Content-Length:".len..], " \t"), 10) catch null;
    }
    return null;
}
