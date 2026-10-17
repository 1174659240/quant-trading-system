#!/usr/bin/env node
// 插件包安全体检：扫「外联主机 / 明文 http / 环境变量读取 / 危险调用 / 本地端口」五类信号。
//
// 用法（在工程根目录）：
//   node tools/audit_plugins.mjs                      # 审默认的量化相关插件
//   node tools/audit_plugins.mjs my-app dsh-stock-watch
//   node tools/audit_plugins.mjs --json               # 只输出 JSON
//
// 判定口径（本项目采用）：出现「明文 http 外联」或「读凭证 + 联网」的组合就不装；
// 只有功能性副作用（起本地服务、建 venv、注入提示词）交由人工权衡。
//
// 依赖：node >= 18，无第三方包。默认去 DSH profile 的 node_modules 找包，
// 并用 realpath 解析 pnpm 的符号链接（否则会扫到 0 个文件）。

import fs from "node:fs";
import path from "node:path";
import os from "node:os";

const DEFAULT_PACKAGES = [
  "@v587d/capital-generation",
  "dsh-fund-research",
  "@ftshare-lab/dsh-kline",
  "dsh-stock-watch",
  "dsh-bayes-predict",
];

const argv = process.argv.slice(2);
const jsonOnly = argv.includes("--json");
const packages = argv.filter((a) => !a.startsWith("--"));
const targets = packages.length ? packages : DEFAULT_PACKAGES;

const PROFILE_DIR =
  process.env.DSH_PROFILE_DIR ||
  path.join(process.env.USERPROFILE || os.homedir(), ".dsh", "profiles", "desktop");
const NODE_MODULES = path.join(PROFILE_DIR, "node_modules");

const SKIP_DIRS = /(^|[\\/])(node_modules|vendor|\.git|dist|build|screenshots)([\\/]|$)/;
const EXTS = new Set([".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json", ".yml", ".yaml"]);
const HOST_RE = /https?:\/\/[A-Za-z0-9._~:/?#[\]@!$&'()*+,;=%-]+/g;
const ENV_RE = /process\.env\.([A-Z0-9_]+)/g;
// 危险调用：注意排除 `regex.exec()` / `obj.spawn` 这类误命中（用否定后视）
const DANGER_RE =
  /\bchild_process\b|(?<!\.)\bexecSync\b|(?<!\.)\bspawnSync\b|(?<!\.)\bspawn\(|(?<!\.)\bexec\(|(?<!\.)\beval\(|new Function|createRequire|writeFileSync|listen\(/g;
// 合法的域名形状（挡掉 URL 正则在 JS 字符串里切出来的碎片，如 `localhost').pathname`）
const HOSTNAME_RE = /^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$/i;
const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost"]);
// 常见的「不是网络外联」的 URL 用途（XML 命名空间、schema 标识、徽章、文档链接）
const BENIGN_HOSTS = new Set([
  "json-schema.org",
  "www.w3.org",
  "semver.org",
  "opensource.org",
  "creativecommons.org",
  "img.shields.io",
  "www.npmjs.com",
  "github.com",
  "example.com",
  "schema.org",
]);

function walk(dir, base = dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (SKIP_DIRS.test(path.relative(base, full))) continue;
    if (entry.isDirectory()) walk(full, base, out);
    else if (EXTS.has(path.extname(entry.name).toLowerCase())) out.push(full);
  }
  return out;
}

function bump(map, key, sample) {
  if (map.has(key)) map.get(key).n += 1;
  else map.set(key, { n: 1, sample });
}

function top(map, limit = 14) {
  return [...map.entries()]
    .sort((a, b) => b[1].n - a[1].n)
    .slice(0, limit)
    .map(([k, v]) => `${k}(${v.n})`);
}

function audit(name) {
  const linkPath = path.join(NODE_MODULES, name);
  if (!fs.existsSync(linkPath)) return { missing: true, looked: linkPath };
  const dir = fs.realpathSync(linkPath);
  const files = walk(dir);
  const https = new Map();
  const plain = new Map();
  const localPorts = new Map();
  const envs = new Map();
  const danger = new Map();
  for (const file of files) {
    let text;
    try {
      text = fs.readFileSync(file, "utf8");
    } catch {
      continue;
    }
    for (const raw of text.match(HOST_RE) ?? []) {
      let url;
      try {
        url = new URL(raw);
      } catch {
        continue;
      }
      if (!HOSTNAME_RE.test(url.hostname)) continue; // 字符串碎片，不是真主机名
      if (BENIGN_HOSTS.has(url.hostname)) continue;
      const bucket = url.protocol === "https:" ? https : plain;
      bump(bucket, url.hostname, raw.slice(0, 90));
    }
    for (const m of text.matchAll(/(?:https?:\/\/)?(?:127\.0\.0\.1|localhost):(\d{2,5})/g)) {
      bump(localPorts, m[1], m[0]);
    }
    for (const m of text.matchAll(ENV_RE)) bump(envs, m[1], m[0]);
    for (const m of text.matchAll(DANGER_RE)) bump(danger, m[0], m[0]);
  }
  const nonLocalPlain = new Map(
    [...plain].filter(([host]) => !LOCAL_HOSTS.has(host)),
  );
  const suspectEnv = [...envs.keys()].filter((v) =>
    /(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/i.test(v),
  );
  return {
    package: name,
    realPath: dir,
    files: files.length,
    httpsHosts: [...https.keys()].filter((h) => !BENIGN_HOSTS.has(h)).length,
    httpsTop: top(https),
    plainHttpNonLocal: top(nonLocalPlain),
    localPorts: top(localPorts),
    envVars: top(envs),
    credentialLikeEnvVars: suspectEnv,
    danger: top(danger),
    verdict:
      nonLocalPlain.size > 0
        ? "REJECT: 明文 http 外联"
        : suspectEnv.length > 0
          ? "REVIEW: 读取疑似凭证的环境变量 + 联网"
          : "PASS(静态扫描)",
  };
}

const report = targets.map(audit);
if (jsonOnly) {
  console.log(JSON.stringify(report, null, 2));
} else {
  for (const r of report) {
    if (r.missing) {
      console.log(`===== ${r.package} =====\n  未安装（找过 ${r.looked}）`);
      continue;
    }
    console.log(`===== ${r.package} =====  ${r.verdict}`);
    console.log(`  文件数 ${r.files} · 有效 https 主机 ${r.httpsHosts}`);
    console.log(`  明文 http（非本机）：${r.plainHttpNonLocal.join(", ") || "无"}`);
    console.log(`  本地端口：${r.localPorts.join(", ") || "无"}`);
    console.log(`  环境变量：${r.envVars.join(", ") || "无"}`);
    console.log(`  疑似凭证变量：${r.credentialLikeEnvVars.join(", ") || "无"}`);
    console.log(`  危险调用：${r.danger.join(", ") || "无"}`);
    console.log(`  路径：${r.realPath}`);
  }
}
