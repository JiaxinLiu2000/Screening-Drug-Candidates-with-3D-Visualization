/**
 * Affinity against toxicity risk, with the applicability domain drawn in.
 *
 * The design sketch called for Plotly. A hand-rolled SVG is used instead: the
 * only interactions needed are hover and rectangle selection, and after the
 * build machine ran out of disk space mid-install, a dependency-free plot was
 * worth more than a configurable one.
 *
 * The substantive change from the sketch is that out-of-domain points are drawn
 * as hollow rings rather than shaded like the rest. Marking only an "ideal
 * region" in the lower right would recommend exactly the molecules the model
 * cannot score: the cold-target Pearson r is 0.48, and the nearest-neighbour
 * distance is the only signal that detects those cases.
 */

import { useMemo, useRef, useState } from "react";
import type { ScreenRow } from "./api";

interface Props {
  rows: ScreenRow[];
  hasAffinity: boolean;
  selected: Set<number>;
  onSelect: (indices: Set<number>) => void;
  onHover: (row: ScreenRow | null) => void;
}

const WIDTH = 620;
const HEIGHT = 380;
const PAD = { top: 18, right: 18, bottom: 46, left: 56 };

/**
 * Where real KIBA affinities actually sit, measured on the 11,720 held-out
 * pairs of `splits/kiba_random_seed0.csv`: median 11.50, interquartile range
 * 11.20-11.95, and 74.7% of values within +-0.6 of the median. The nominal
 * 0.0-17.2 range comes from rare extremes and makes the target look far more
 * spread out than it is.
 */
const KIBA_REFERENCE = { low: 11.2, median: 11.5, high: 11.95 };

interface Point {
  row: ScreenRow;
  x: number;
  y: number;
  inDomain: boolean;
  qed: number;
}

export function ScatterPlot({ rows, hasAffinity, selected, onSelect, onHover }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(
    null,
  );

  const points = useMemo<Point[]>(() => {
    const out: Point[] = [];
    for (const row of rows) {
      if (row.error) continue;
      const y = row.toxicity_risk_max;
      const x = hasAffinity ? row.affinity_kiba_score : row.descriptors.clogp;
      if (x === null || y === null) continue;
      out.push({
        row,
        x,
        y,
        inDomain: row.applicability_toxicity.in_domain,
        qed: row.descriptors.qed ?? 0.3,
      });
    }
    return out;
  }, [rows, hasAffinity]);

  const scales = useMemo(() => {
    if (points.length === 0) return null;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    let xMin = Math.min(...xs);
    let xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);

    // Auto-scaling the affinity axis to the batch is actively misleading here.
    // Three quarters of real KIBA values sit within +-0.6 of the 11.50 median
    // (interquartile range 11.20-11.95), so a batch spanning 1.4 units is a
    // *normal* spread -- but stretched across the full plot width it reads as a
    // dramatic separation. The axis is therefore widened to always contain the
    // reference band, so the batch is seen against the distribution it came
    // from rather than against itself.
    if (hasAffinity) {
      xMin = Math.min(xMin, KIBA_REFERENCE.low - 0.3);
      xMax = Math.max(xMax, KIBA_REFERENCE.high + 0.3);
    }

    // A single point, or a column of identical values, would give a zero span
    // and divide by zero; widen it instead of dropping the data.
    const xSpan = xMax - xMin || 1;
    const ySpan = yMax - yMin || 1;
    const plotW = WIDTH - PAD.left - PAD.right;
    const plotH = HEIGHT - PAD.top - PAD.bottom;
    return {
      xMin,
      xMax,
      yMin,
      yMax,
      sx: (v: number) => PAD.left + ((v - xMin) / xSpan) * plotW,
      sy: (v: number) => PAD.top + plotH - ((v - yMin) / ySpan) * plotH,
    };
  }, [points, hasAffinity]);

  if (!scales) {
    return (
      <div className="plot-empty">
        Nothing to plot yet — submit molecules to see them placed on both axes.
      </div>
    );
  }

  const toLocal = (event: React.MouseEvent) => {
    const box = svgRef.current!.getBoundingClientRect();
    return {
      x: ((event.clientX - box.left) / box.width) * WIDTH,
      y: ((event.clientY - box.top) / box.height) * HEIGHT,
    };
  };

  const finishDrag = () => {
    if (!drag) return;
    const xLo = Math.min(drag.x0, drag.x1);
    const xHi = Math.max(drag.x0, drag.x1);
    const yLo = Math.min(drag.y0, drag.y1);
    const yHi = Math.max(drag.y0, drag.y1);
    if (xHi - xLo < 4 && yHi - yLo < 4) {
      onSelect(new Set());
    } else {
      const hit = new Set<number>();
      for (const p of points) {
        const px = scales.sx(p.x);
        const py = scales.sy(p.y);
        if (px >= xLo && px <= xHi && py >= yLo && py <= yHi) {
          hit.add(p.row.input_index);
        }
      }
      onSelect(hit);
    }
    setDrag(null);
  };

  const xLabel = hasAffinity
    ? "Predicted affinity (KIBA score, higher binds tighter) — not pKd"
    : "cLogP (no target selected, so no affinity axis)";

  const ticks = (lo: number, hi: number) =>
    Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);

  return (
    <svg
      ref={svgRef}
      className="scatter"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      onMouseDown={(e) => {
        const { x, y } = toLocal(e);
        setDrag({ x0: x, y0: y, x1: x, y1: y });
      }}
      onMouseMove={(e) => {
        if (!drag) return;
        const { x, y } = toLocal(e);
        setDrag({ ...drag, x1: x, y1: y });
      }}
      onMouseUp={finishDrag}
      onMouseLeave={() => {
        if (drag) finishDrag();
        onHover(null);
      }}
    >
      <rect
        x={PAD.left}
        y={PAD.top}
        width={WIDTH - PAD.left - PAD.right}
        height={HEIGHT - PAD.top - PAD.bottom}
        className="plot-frame"
      />

      {/* The band where half of all measured KIBA affinities fall, plus the
          median. Without it a viewer has no way to tell whether the batch is
          genuinely spread out or packed into the middle of the distribution. */}
      {hasAffinity && (
        <g>
          <rect
            x={scales.sx(KIBA_REFERENCE.low)}
            y={PAD.top}
            width={scales.sx(KIBA_REFERENCE.high) - scales.sx(KIBA_REFERENCE.low)}
            height={HEIGHT - PAD.top - PAD.bottom}
            className="ref-band"
          />
          <line
            x1={scales.sx(KIBA_REFERENCE.median)}
            x2={scales.sx(KIBA_REFERENCE.median)}
            y1={PAD.top}
            y2={HEIGHT - PAD.bottom}
            className="ref-median"
          />
          <text
            x={scales.sx(KIBA_REFERENCE.high) + 6}
            y={PAD.top + 11}
            className="ref-label"
          >
            half of all measured values
          </text>
        </g>
      )}

      {ticks(scales.xMin, scales.xMax).map((v) => (
        <g key={`x${v}`}>
          <line
            x1={scales.sx(v)}
            x2={scales.sx(v)}
            y1={PAD.top}
            y2={HEIGHT - PAD.bottom}
            className="grid"
          />
          <text x={scales.sx(v)} y={HEIGHT - PAD.bottom + 16} className="tick">
            {v.toFixed(1)}
          </text>
        </g>
      ))}
      {ticks(scales.yMin, scales.yMax).map((v) => (
        <g key={`y${v}`}>
          <line
            x1={PAD.left}
            x2={WIDTH - PAD.right}
            y1={scales.sy(v)}
            y2={scales.sy(v)}
            className="grid"
          />
          <text x={PAD.left - 8} y={scales.sy(v) + 4} className="tick tick-y">
            {v.toFixed(2)}
          </text>
        </g>
      ))}

      {points.map((p) => {
        const isSelected = selected.has(p.row.input_index);
        const radius = 4 + p.qed * 7;
        return (
          <circle
            key={p.row.input_index}
            cx={scales.sx(p.x)}
            cy={scales.sy(p.y)}
            r={radius}
            className={[
              "point",
              p.inDomain ? "in-domain" : "out-domain",
              isSelected ? "selected" : "",
            ]
              .filter(Boolean)
              .join(" ")}
            onMouseEnter={() => onHover(p.row)}
          />
        );
      })}

      {drag && (
        <rect
          x={Math.min(drag.x0, drag.x1)}
          y={Math.min(drag.y0, drag.y1)}
          width={Math.abs(drag.x1 - drag.x0)}
          height={Math.abs(drag.y1 - drag.y0)}
          className="brush"
        />
      )}

      <text x={WIDTH / 2} y={HEIGHT - 8} className="axis-label">
        {xLabel}
      </text>
      <text
        x={-(HEIGHT / 2)}
        y={14}
        transform="rotate(-90)"
        className="axis-label"
      >
        Toxicity risk (max of 12 endpoints)
      </text>
    </svg>
  );
}
