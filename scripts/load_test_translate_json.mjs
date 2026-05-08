#!/usr/bin/env node
// Heavy load test cho endpoint POST /v1/json (translate batch)
// Zero-dependency, chạy trên Node >= 18 (đã test trên v25).
//
// Ví dụ:
//   node scripts/load_test_translate_json.mjs \
//        --url http://89.221.67.144:9000 \
//        --concurrency 32 --duration 60 \
//        --batch-size 10 --target-lang vi
//
//   node scripts/load_test_translate_json.mjs \
//        --url http://localhost:9000 \
//        --concurrency 16 --total 2000 --batch-size 5
//
// In đầy đủ p50/p90/p95/p99, RPS, error breakdown.

import { parseArgs } from "node:util";
import { performance } from "node:perf_hooks";
import { writeFileSync } from "node:fs";

const DEFAULT_CORPUS = [
  "The quick brown fox jumps over the lazy dog.",
  "Artificial intelligence is reshaping how software is built and shipped.",
  "Please translate the following paragraph into Vietnamese as accurately as possible.",
  "She sells seashells by the seashore on a sunny Sunday morning.",
  "Climate change requires coordinated action from governments and industries worldwide.",
  "Machine translation has improved dramatically thanks to large language models.",
  "Привет, как дела сегодня? Надеюсь, что хорошо.",
  "今日は天気がとても良いので、公園に散歩に行きましょう。",
  "안녕하세요, 오늘 회의는 오후 세 시에 시작됩니다.",
  "La inteligencia artificial está cambiando la forma en que trabajamos.",
  "Bonjour tout le monde, ravi de vous rencontrer aujourd'hui.",
  "Guten Morgen! Wie war Ihr Wochenende?",
  "Đây là một câu tiếng Việt có dấu để kiểm tra mô hình dịch ngược lại.",
  "Quantum computing promises to solve problems that classical computers cannot.",
  "The committee agreed to postpone the decision until next quarter.",
  "Renewable energy sources now generate over thirty percent of global electricity.",
  "Open source software powers a significant portion of modern internet infrastructure.",
  "Effective communication is the cornerstone of successful team collaboration.",
  "She finished her marathon training despite the unusually hot weather.",
  "Cybersecurity threats continue to evolve faster than defensive technologies.",
  "中文翻译质量在过去几年取得了显著进步。",
  "Tiếng Việt có dấu là yêu cầu bắt buộc trong mọi tài liệu nội bộ.",
  "Berlin, Paris and Tokyo are popular destinations for international travelers.",
  "Reading books regularly improves vocabulary and critical thinking skills.",
  "The project deadline was extended by two weeks due to unforeseen issues.",
];

const { values: args } = parseArgs({
  options: {
    url: { type: "string", default: "http://localhost:9000" },
    concurrency: { type: "string", default: "16" },
    duration: { type: "string", default: "0" }, // giây; 0 = không dùng
    total: { type: "string", default: "0" },     // tổng request; 0 = không dùng
    "batch-size": { type: "string", default: "5" },
    "source-lang": { type: "string", default: "auto" },
    "target-lang": { type: "string", default: "vi" },
    timeout: { type: "string", default: "60000" }, // ms
    warmup: { type: "string", default: "5" },
    "report-json": { type: "string", default: "" }, // path tuỳ chọn dump JSON
    corpus: { type: "string", default: "" },         // path file JSON array<string>
    quiet: { type: "boolean", default: false },
    help: { type: "boolean", default: false },
  },
  allowPositionals: false,
});

if (args.help) {
  console.log(`Heavy load test cho POST /v1/json

Cờ:
  --url <base>             Base URL của API (vd http://localhost:9000)
  --concurrency <N>        Số virtual user song song (default 16)
  --duration <sec>         Chạy theo thời gian (giây). 0 = bỏ qua.
  --total <N>              Tổng số request (thay cho duration)
  --batch-size <N>         Số texts mỗi request (1-100, default 5)
  --source-lang <code>     auto | vi | en | ja | zh ... (default auto)
  --target-lang <code>     ngôn ngữ đích, bắt buộc khác auto (default vi)
  --timeout <ms>           Per-request timeout, default 60000
  --warmup <N>             Số request warmup không tính vào metrics (default 5)
  --corpus <path>          File JSON chứa array string (override default)
  --report-json <path>     Ghi metrics ra file JSON
  --quiet                  Tắt log progress mỗi giây
  --help                   In help này
`);
  process.exit(0);
}

const baseUrl = args.url.replace(/\/+$/, "");
const concurrency = Math.max(1, parseInt(args.concurrency, 10));
const durationSec = Math.max(0, parseInt(args.duration, 10));
const totalRequests = Math.max(0, parseInt(args.total, 10));
const batchSize = Math.min(100, Math.max(1, parseInt(args["batch-size"], 10)));
const sourceLang = args["source-lang"];
const targetLang = args["target-lang"];
const timeoutMs = Math.max(1000, parseInt(args.timeout, 10));
const warmupCount = Math.max(0, parseInt(args.warmup, 10));
const reportJsonPath = args["report-json"];
const quiet = args.quiet;

if (!durationSec && !totalRequests) {
  console.error("Phải truyền một trong hai: --duration hoặc --total");
  process.exit(2);
}
if (targetLang === "auto") {
  console.error("--target-lang không được là 'auto'");
  process.exit(2);
}

let corpus = DEFAULT_CORPUS;
if (args.corpus) {
  const raw = await import("node:fs").then((fs) =>
    fs.readFileSync(args.corpus, "utf8"),
  );
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed) || parsed.length === 0) {
    console.error("--corpus phải là file JSON chứa mảng string không rỗng");
    process.exit(2);
  }
  corpus = parsed.filter((s) => typeof s === "string" && s.trim().length > 0);
  if (corpus.length === 0) {
    console.error("Corpus rỗng sau khi lọc");
    process.exit(2);
  }
}

const endpoint = `${baseUrl}/v1/json`;

function buildPayload() {
  const texts = [];
  for (let i = 0; i < batchSize; i++) {
    texts.push(corpus[Math.floor(Math.random() * corpus.length)]);
  }
  return {
    texts,
    source_lang: sourceLang,
    target_lang: targetLang,
  };
}

class Stats {
  constructor() {
    this.latencies = []; // ms — chỉ tính request thành công
    this.success = 0;
    this.fail = 0;
    this.totalTextsTranslated = 0;
    this.bytesIn = 0;
    this.bytesOut = 0;
    this.errorsByKind = new Map();
    this.serverProcessingMs = []; // processing_time_ms từ server
    this.startedAt = 0;
    this.endedAt = 0;
  }
  recordSuccess(clientMs, serverMs, textsCount, bytesIn, bytesOut) {
    this.success += 1;
    this.latencies.push(clientMs);
    if (typeof serverMs === "number") this.serverProcessingMs.push(serverMs);
    this.totalTextsTranslated += textsCount;
    this.bytesIn += bytesIn;
    this.bytesOut += bytesOut;
  }
  recordFailure(kind) {
    this.fail += 1;
    this.errorsByKind.set(kind, (this.errorsByKind.get(kind) ?? 0) + 1);
  }
  total() {
    return this.success + this.fail;
  }
}

function percentile(sortedArr, p) {
  if (sortedArr.length === 0) return 0;
  const idx = Math.min(
    sortedArr.length - 1,
    Math.floor((p / 100) * sortedArr.length),
  );
  return sortedArr[idx];
}

function summarize(latencies) {
  if (latencies.length === 0) {
    return { count: 0, min: 0, max: 0, avg: 0, p50: 0, p90: 0, p95: 0, p99: 0 };
  }
  const sorted = [...latencies].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    count: sorted.length,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    avg: sum / sorted.length,
    p50: percentile(sorted, 50),
    p90: percentile(sorted, 90),
    p95: percentile(sorted, 95),
    p99: percentile(sorted, 99),
  };
}

async function sendOne(stats, recordToStats) {
  const payload = buildPayload();
  const body = JSON.stringify(payload);
  const bytesOut = Buffer.byteLength(body, "utf8");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const t0 = performance.now();
  let res;
  try {
    res = await fetch(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body,
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    if (recordToStats) {
      const kind =
        err?.name === "AbortError" ? "timeout" : `network:${err?.code ?? err?.message ?? "unknown"}`;
      stats.recordFailure(kind);
    }
    return;
  }
  let textBody;
  try {
    textBody = await res.text();
  } catch (err) {
    clearTimeout(timer);
    if (recordToStats) stats.recordFailure(`read_body:${err?.message ?? "unknown"}`);
    return;
  }
  clearTimeout(timer);
  const clientMs = performance.now() - t0;
  const bytesIn = Buffer.byteLength(textBody, "utf8");

  if (!res.ok) {
    if (recordToStats) {
      stats.recordFailure(`http_${res.status}`);
    }
    return;
  }

  let parsed;
  try {
    parsed = JSON.parse(textBody);
  } catch {
    if (recordToStats) stats.recordFailure("invalid_json");
    return;
  }

  const translations = parsed?.translations;
  if (!Array.isArray(translations) || translations.length !== payload.texts.length) {
    if (recordToStats) stats.recordFailure("shape_mismatch");
    return;
  }
  if (recordToStats) {
    stats.recordSuccess(
      clientMs,
      typeof parsed.processing_time_ms === "number" ? parsed.processing_time_ms : null,
      translations.length,
      bytesIn,
      bytesOut,
    );
  }
}

async function warmup() {
  if (warmupCount === 0) return;
  if (!quiet) console.log(`[warmup] gửi ${warmupCount} request (không tính vào metrics)...`);
  const dummyStats = new Stats();
  const tasks = [];
  for (let i = 0; i < warmupCount; i++) tasks.push(sendOne(dummyStats, false));
  await Promise.all(tasks);
}

function fmt(n, digits = 1) {
  if (!isFinite(n)) return "n/a";
  return n.toFixed(digits);
}

function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function printProgress(stats) {
  const elapsed = (performance.now() - stats.startedAt) / 1000;
  const total = stats.total();
  const rps = elapsed > 0 ? total / elapsed : 0;
  const recent = summarize(stats.latencies.slice(-200));
  process.stdout.write(
    `\r[${fmt(elapsed)}s] req=${total} ok=${stats.success} fail=${stats.fail} ` +
      `rps=${fmt(rps)} p50=${fmt(recent.p50)}ms p95=${fmt(recent.p95)}ms      `,
  );
}

async function runWorker(stats, stopWhen) {
  while (!stopWhen()) {
    await sendOne(stats, true);
  }
}

async function main() {
  console.log(`Heavy load test → ${endpoint}`);
  console.log(
    `concurrency=${concurrency} batchSize=${batchSize} source=${sourceLang} target=${targetLang} ` +
      (durationSec ? `duration=${durationSec}s` : `total=${totalRequests}`),
  );

  // Probe nhanh để fail-fast nếu URL sai
  try {
    const probe = await fetch(`${baseUrl}/v1/health`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!probe.ok && probe.status !== 404) {
      console.warn(`[probe] /v1/health trả ${probe.status} — vẫn chạy load test`);
    }
  } catch (err) {
    console.warn(`[probe] không reach được ${baseUrl}: ${err?.message ?? err} — vẫn thử chạy`);
  }

  await warmup();

  const stats = new Stats();
  stats.startedAt = performance.now();

  // Điều kiện dừng
  let issued = 0;
  const useTotal = totalRequests > 0;
  const deadline = durationSec > 0 ? stats.startedAt + durationSec * 1000 : Infinity;
  const stopWhen = () => {
    if (useTotal) return issued >= totalRequests;
    return performance.now() >= deadline;
  };

  // Mỗi worker tự lấy slot. Với mode total: dùng counter atomic.
  async function workerTotal() {
    while (true) {
      if (issued >= totalRequests) return;
      issued += 1;
      await sendOne(stats, true);
    }
  }
  async function workerDuration() {
    while (performance.now() < deadline) {
      await sendOne(stats, true);
    }
  }

  // Progress ticker
  let progressTimer = null;
  if (!quiet) {
    progressTimer = setInterval(() => printProgress(stats), 1000);
  }

  // SIGINT để dừng sớm
  let interrupted = false;
  process.on("SIGINT", () => {
    interrupted = true;
    console.log("\n[!] SIGINT — đang dừng worker...");
    if (useTotal) issued = totalRequests; // chặn issue mới
    // duration mode: deadline đã set, ép luôn về now
  });

  const workers = [];
  for (let i = 0; i < concurrency; i++) {
    workers.push(useTotal ? workerTotal() : workerDuration());
  }
  await Promise.all(workers);

  stats.endedAt = performance.now();
  if (progressTimer) clearInterval(progressTimer);
  if (!quiet) {
    printProgress(stats);
    process.stdout.write("\n");
  }

  // ===== Báo cáo =====
  const elapsedSec = (stats.endedAt - stats.startedAt) / 1000;
  const total = stats.total();
  const rps = total / elapsedSec;
  const successRate = total > 0 ? (stats.success / total) * 100 : 0;
  const lat = summarize(stats.latencies);
  const srv = summarize(stats.serverProcessingMs);
  const textsPerSec = stats.totalTextsTranslated / elapsedSec;

  const errorRows = [...stats.errorsByKind.entries()].sort((a, b) => b[1] - a[1]);

  console.log("\n========== KẾT QUẢ ==========");
  console.log(`Endpoint           : ${endpoint}`);
  console.log(`Concurrency        : ${concurrency}`);
  console.log(`Batch size         : ${batchSize} texts/request`);
  console.log(`Thời gian chạy     : ${fmt(elapsedSec)} s${interrupted ? " (interrupt)" : ""}`);
  console.log(`Tổng request       : ${total} (ok=${stats.success}, fail=${stats.fail})`);
  console.log(`Success rate       : ${fmt(successRate, 2)} %`);
  console.log(`Throughput         : ${fmt(rps, 2)} req/s, ${fmt(textsPerSec, 2)} texts/s`);
  console.log(`Bytes in / out     : ${fmtBytes(stats.bytesIn)} / ${fmtBytes(stats.bytesOut)}`);
  console.log("");
  console.log("Latency client (ms):");
  console.log(
    `  min=${fmt(lat.min)}  avg=${fmt(lat.avg)}  p50=${fmt(lat.p50)}  ` +
      `p90=${fmt(lat.p90)}  p95=${fmt(lat.p95)}  p99=${fmt(lat.p99)}  max=${fmt(lat.max)}`,
  );
  if (srv.count > 0) {
    console.log("Server processing_time_ms:");
    console.log(
      `  min=${fmt(srv.min)}  avg=${fmt(srv.avg)}  p50=${fmt(srv.p50)}  ` +
        `p90=${fmt(srv.p90)}  p95=${fmt(srv.p95)}  p99=${fmt(srv.p99)}  max=${fmt(srv.max)}`,
    );
  }
  if (errorRows.length > 0) {
    console.log("\nLỗi theo loại:");
    for (const [kind, count] of errorRows) {
      console.log(`  ${count.toString().padStart(6)}  ${kind}`);
    }
  }
  console.log("==============================");

  if (reportJsonPath) {
    const report = {
      endpoint,
      config: {
        concurrency,
        batchSize,
        durationSec: durationSec || null,
        totalRequests: totalRequests || null,
        sourceLang,
        targetLang,
        timeoutMs,
        warmupCount,
      },
      summary: {
        elapsedSec,
        totalRequests: total,
        success: stats.success,
        fail: stats.fail,
        successRate,
        rps,
        textsPerSec,
        bytesIn: stats.bytesIn,
        bytesOut: stats.bytesOut,
        interrupted,
      },
      latencyClientMs: lat,
      serverProcessingMs: srv,
      errors: Object.fromEntries(errorRows),
      generatedAt: new Date().toISOString(),
    };
    writeFileSync(reportJsonPath, JSON.stringify(report, null, 2));
    console.log(`Đã ghi report JSON: ${reportJsonPath}`);
  }

  // Exit code: 1 nếu có fail, 0 nếu sạch
  process.exit(stats.fail > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("\nLỗi không bắt được:", err);
  process.exit(1);
});
