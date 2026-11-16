# Browser demo

`index.html` is a **self-contained** demo of the pure-JS (`js/`) library: no
build step, no network, no dependencies. Open it in any modern browser and you
can generate a tone WAV (or upload audio), sign it with Content Credentials,
inspect the manifest, verify it, and flip a single byte to watch the hard
binding break — all client-side via WebCrypto (ES256 / ECDSA P-256).

## Regenerating

`index.html` is generated from the library source so it can never drift. It
inlines the exact bytes of `js/default-cert.mjs`, `js/c2pa.mjs`, and
`js/c2pa-verify.mjs` (each wrapped in a block scope so their top-level helpers
don't collide) and hoists their exports onto a `LIB` object the page drives.

```bash
node demo/build.mjs        # rewrites demo/index.html from js/*
```

Run it whenever the library changes. The signing cert is the repo's bundled
default (`js/default-cert.mjs`) — a demo identity, not a trust anchor.
