import { readFileSync, writeFileSync } from "node:fs";
import { instance } from "@viz-js/viz";

const [, , inputPath, outputPath] = process.argv;

if (!inputPath || !outputPath) {
  console.error("usage: node scripts/render_graphviz_svg.mjs input.dot output.svg");
  process.exit(1);
}

const viz = await instance();
const dot = readFileSync(inputPath, "utf8");
const svg = viz.renderString(dot, { format: "svg", engine: "dot" });
writeFileSync(outputPath, svg);
