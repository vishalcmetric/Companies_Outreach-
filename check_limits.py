import sys, json, urllib.request, urllib.error, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
KEY = sys.argv[1] if len(sys.argv) > 1 else ""
if not KEY:
    print("Usage: python check_limits.py gsk_YOUR_KEY")
    sys.exit(1)

BASE = "https://api.groq.com/openai/v1"

def call(messages, max_tokens=10):
    body = json.dumps({"model":"llama-3.3-70b-versatile","max_tokens":max_tokens,"messages":messages}).encode()
    req = urllib.request.Request(BASE+"/chat/completions", data=body,
        headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY,"User-Agent":"Mozilla/5.0"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read()), dict(r.getheaders())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace"), {}

print("Firing 3 quick calls to measure token consumption per call...")
for i in range(1, 4):
    t0 = time.time()
    status, data, hdrs = call([{"role":"user","content":"Say OK in one word"}])
    elapsed = time.time()-t0
    if status == 200:
        u = data.get("usage",{})
        tpm_lim = hdrs.get("x-ratelimit-limit-tokens","?")
        tpm_rem = hdrs.get("x-ratelimit-remaining-tokens","?")
        rpm_lim = hdrs.get("x-ratelimit-limit-requests","?")
        rpm_rem = hdrs.get("x-ratelimit-remaining-requests","?")
        print(f"Call {i}: OK in {elapsed:.1f}s | tokens={u.get('total_tokens','?')} | TPM limit={tpm_lim} remaining={tpm_rem} | RPM limit={rpm_lim} remaining={rpm_rem}")
    elif status == 429:
        print(f"Call {i}: 429 RATE LIMITED -> {str(data)[:200]}")
        break
    else:
        print(f"Call {i}: HTTP {status} -> {str(data)[:200]}")
        break
    time.sleep(0.5)
