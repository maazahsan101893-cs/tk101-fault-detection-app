import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { FaultEvent, fmtTs, ModelResult } from "../api";
import { tooltipStyle } from "./chartTheme";

interface Props {
  result: ModelResult;
  color: string;
  logScale: boolean;
  height?: number;
  selected?: FaultEvent | null;
  compact?: boolean;
}

const FLOOR = 1e-4;

export default function ScoreChart({ result, color, logScale, height = 320, selected, compact }: Props) {
  const { data, yMax } = useMemo(() => {
    const max = Math.max(result.threshold * 1.5, ...result.series.map((p) => p.score));
    return {
      yMax: max,
      data: result.series.map((p) => ({
        t: p.t,
        score: logScale ? Math.max(p.score, FLOOR) : p.score,
        band: p.event ? max * (logScale ? 1.2 : 1.05) : null,
      })),
    };
  }, [result, logScale]);

  const highlight = useMemo(() => {
    if (!selected || !data.length) return null;
    let i1 = data.findIndex((d) => d.t >= selected.start);
    if (i1 < 0) i1 = data.length - 1;
    let i2 = i1;
    while (i2 + 1 < data.length && data[i2 + 1].t <= selected.end) i2++;
    if (i1 > 0 && data[i1].t > selected.start) i1--;
    return { x1: data[i1].t, x2: data[i2].t };
  }, [selected, data]);

  const yMin = logScale ? Math.max(FLOOR, Math.min(...data.map((d) => d.score)) * 0.8) : 0;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, left: compact ? -10 : 4, bottom: 0 }}>
        <CartesianGrid stroke="#1c2633" vertical={false} />
        <XAxis
          dataKey="t"
          tickFormatter={(v: string) => fmtTs(v)}
          minTickGap={compact ? 60 : 80}
          tickLine={false}
          axisLine={{ stroke: "#243041" }}
        />
        <YAxis
          scale={logScale ? "log" : "linear"}
          domain={[yMin, yMax * (logScale ? 1.3 : 1.08)]}
          allowDataOverflow
          tickFormatter={(v: number) => (v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toPrecision(2))}
          width={compact ? 44 : 52}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip
          {...tooltipStyle}
          labelFormatter={(v: string) => fmtTs(v, true) + " UTC"}
          formatter={(v: number, name: string) => (name === "band" ? ["in fault event", "status"] : [v.toFixed(4), "anomaly score"])}
        />
        <Area
          dataKey="band"
          type="stepAfter"
          stroke="none"
          fill="rgba(248,113,113,0.22)"
          isAnimationActive={false}
          connectNulls={false}
          baseValue={yMin}
          activeDot={false}
        />
        {highlight && (
          <ReferenceArea x1={highlight.x1} x2={highlight.x2} fill="rgba(245,158,11,0.25)" stroke="#f59e0b" strokeOpacity={0.6} />
        )}
        <Line dataKey="score" stroke={color} dot={false} strokeWidth={1.2} isAnimationActive={false} />
        <ReferenceLine
          y={result.threshold}
          stroke="#f87171"
          strokeDasharray="5 4"
          label={compact ? undefined : { value: `99% limit ${result.threshold.toFixed(3)}`, fill: "#f87171", fontSize: 11, position: "insideTopRight" }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
