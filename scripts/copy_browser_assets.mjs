import { cp, mkdir } from "node:fs/promises";

await mkdir("assets/vendor", { recursive: true });
await cp("node_modules/reveal.js/dist", "assets/vendor/reveal", { recursive: true });
await cp("node_modules/reveal.js/plugin", "assets/vendor/reveal/plugin", { recursive: true });
await cp("node_modules/reveal.js/LICENSE", "assets/vendor/reveal/LICENSE");
await cp("node_modules/pdfjs-dist/build", "assets/vendor/pdfjs", { recursive: true });
await cp("node_modules/pdfjs-dist/standard_fonts", "assets/vendor/standard_fonts", { recursive: true });
await cp("node_modules/pdfjs-dist/cmaps", "assets/vendor/cmaps", { recursive: true });
await cp("node_modules/pdfjs-dist/LICENSE", "assets/vendor/pdfjs/LICENSE");
await cp("node_modules/requirejs/require.js", "assets/vendor/require.js");
await cp("node_modules/mathjax", "assets/vendor/mathjax", { recursive: true });
