import React, { useRef, useMemo } from "react";
import ForceGraph2D from "react-force-graph-2d";

/**
 * ConnectionGraph
 *
 * Props:
 *  - connections: array of { source, target, relation? }
 *  - edges: alternative prop name (same shape as connections)
 *  - nodesFromEdges: boolean, if true build nodes from edges automatically (default: true)
 *  - height: CSS height for container (default: 400)
 *
 * This component is defensive: it accepts either `connections` or `edges`,
 * and handles undefined/null inputs safely by treating them as empty arrays.
 */
export default function ConnectionGraph({
  connections,
  edges,
  nodesFromEdges = true,
  height = 400,
}) {
  const fgRef = useRef();

  // Normalize incoming data: prefer `connections`, then `edges`.
  const rawEdges = useMemo(() => {
    const candidate = connections ?? edges ?? [];
    // Ensure we have an array; if not, attempt to coerce to array safely.
    if (!Array.isArray(candidate)) {
      // If it's an object with keys, try to extract values; otherwise fallback to empty.
      if (candidate && typeof candidate === "object") {
        return Object.values(candidate);
      }
      return [];
    }
    return candidate;
  }, [connections, edges]);

  // Build nodes and links safely
  const { nodes, links } = useMemo(() => {
    const nodesMap = new Map();
    const safeLinks = [];

    if (Array.isArray(rawEdges)) {
      rawEdges.forEach((item) => {
        // item may be a string or a simple tuple or an object
        let src = null;
        let tgt = null;
        let rel = undefined;

        if (!item) return;

        if (typeof item === "string") {
          // can't infer source/target from a plain string, skip
          return;
        }

        if (Array.isArray(item)) {
          // [source, target, relation?]
          src = item[0];
          tgt = item[1];
          rel = item[2];
        } else if (typeof item === "object") {
          src = item.source ?? item.from ?? item.src ?? item.a ?? null;
          tgt = item.target ?? item.to ?? item.dst ?? item.b ?? null;
          rel = item.relation ?? item.label ?? item.type ?? item.relation_type;
        }

        if (!src || !tgt) return;

        const s = String(src);
        const t = String(tgt);

        if (!nodesMap.has(s)) nodesMap.set(s, { id: s });
        if (!nodesMap.has(t)) nodesMap.set(t, { id: t });

        safeLinks.push({
          source: s,
          target: t,
          relation: rel || "associated_with",
        });
      });
    }

    return {
      nodes: Array.from(nodesMap.values()),
      links: safeLinks,
    };
  }, [rawEdges]);

  // Tooltip for links (handles both node objects and ids)
  const getLinkTooltip = (link) => {
    try {
      const src =
        typeof link.source === "object" ? link.source.id : link.source;
      const tgt =
        typeof link.target === "object" ? link.target.id : link.target;
      return `${src} ${link.relation || "associated_with"} ${tgt}`;
    } catch {
      return link.relation || "associated_with";
    }
  };

  // Node canvas drawing with label
  const nodeCanvasObject = (node, ctx, globalScale) => {
    const label = node.id || "";
    const fontSize = Math.max(10, 12 / globalScale);
    ctx.font = `${fontSize}px Sans-Serif`;
    ctx.fillStyle = "#0f172a";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    if (typeof node.x === "number" && typeof node.y === "number") {
      ctx.fillText(label, node.x, node.y + fontSize);
    }
  };

  // If there are no links/nodes, render an empty placeholder container
  if ((!nodes || nodes.length === 0) && (!links || links.length === 0)) {
    return (
      <div
        style={{
          height,
          border: "1px dashed #e6e9ee",
          borderRadius: 6,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#6b7280",
          padding: 12,
          boxSizing: "border-box",
          background: "#fff",
        }}
      >
        No connections to display
      </div>
    );
  }

  return (
    <div
      style={{
        height,
        border: "1px solid #e6e9ee",
        borderRadius: 6,
        background: "#fff",
      }}
    >
      <ForceGraph2D
        ref={fgRef}
        graphData={{ nodes: nodesFromEdges ? nodes : [], links }}
        nodeAutoColorBy="id"
        linkDirectionalArrowLength={6}
        linkDirectionalArrowRelPos={1}
        linkDirectionalParticleWidth={2}
        linkWidth={2}
        nodeCanvasObject={nodeCanvasObject}
        linkDirectionalParticles={1}
        linkDirectionalParticleSpeed={0.005}
        linkLabel={getLinkTooltip}
        minZoom={0.2}
        maxZoom={4}
      />
    </div>
  );
}
