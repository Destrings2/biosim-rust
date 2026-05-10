// Tiny SVG illustrations that hint at each challenge's geometry. Used in the
// challenge picker tiles and the challenge panel header. Designs ported from
// the original design-agent prototype — adapted to the real challenge ids
// surfaced by `Simulator.get_challenge_schemas()`.

const stroke = "rgba(125, 211, 168, 0.7)";
const fill   = "rgba(125, 211, 168, 0.15)";
const dim    = "rgba(168, 174, 185, 0.4)";

// Map real challenge ids → art kind. Anything unknown falls through to "free".
const ART_BY_ID: Record<string, string> = {
  circle:              "circle",
  right_half:          "halves",
  right_quarter:       "halves",
  left_eighth:         "halves",
  east_west_eighths:   "split",
  center_weighted:     "circle",
  center_unweighted:   "circle",
  corner:              "corner",
  corner_weighted:     "corner",
  against_any_wall:    "walls",
  near_barrier:        "wall",
  pairs:               "pairs",
  center_sparse:       "density",
  string:              "stripe",
  migrate_distance:    "migrate",
  touch_any_wall:      "walls",
  location_sequence:   "diamond",
  radioactive_walls:   "walls",
  altruism:            "altruism",
  altruism_sacrifice:  "altruism",
  sun_tracker:         "sun",
  diaspora:            "split",
  food_foraging:       "pellets",
  survivor:            "drift",
};

export function challengeArtKind(id: string): string {
  return ART_BY_ID[id] ?? "free";
}

interface ArtProps { kind: string }

export function ChallengeArt({ kind }: ArtProps) {
  switch (kind) {
    case "circle":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/><circle cx="50" cy="30" r="16" fill={fill} stroke={stroke}/><g fill="rgba(255,255,255,0.5)"><circle cx="50" cy="30" r="0.7"/><circle cx="46" cy="32" r="0.7"/><circle cx="54" cy="28" r="0.7"/><circle cx="48" cy="26" r="0.7"/><circle cx="52" cy="33" r="0.7"/></g></svg>;
    case "halves":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/><rect x="50" y="4" width="48" height="52" fill={fill} stroke={stroke}/></svg>;
    case "corner":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/><rect x="2" y="2" width="22" height="22" fill={fill} stroke={stroke}/></svg>;
    case "sun":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/><circle cx="20" cy="30" r="6" fill={fill} stroke={stroke}/><circle cx="50" cy="30" r="6" fill="none" stroke={stroke} opacity={0.5} strokeDasharray="2 2"/><circle cx="80" cy="30" r="6" fill="none" stroke={stroke} opacity={0.25} strokeDasharray="2 2"/><path d="M 26 30 L 74 30" stroke={stroke} strokeDasharray="1 3" opacity={0.5}/></svg>;
    case "walls":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke="rgba(224,123,123,0.6)" strokeWidth="2"/><rect x="14" y="14" width="72" height="32" fill={fill} stroke={stroke}/></svg>;
    case "pellets":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/>{[[20,18],[35,40],[55,16],[70,42],[80,22],[40,30],[60,40]].map((p,i)=>(<circle key={i} cx={p[0]} cy={p[1]} r="2.2" fill={stroke}/>))}</svg>;
    case "migrate":
      return <svg viewBox="0 0 100 60"><defs><marker id="arr" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill={stroke}/></marker></defs><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/><rect x="2" y="2" width="14" height="56" fill="rgba(168,174,185,0.15)" stroke={dim}/><rect x="84" y="2" width="14" height="56" fill={fill} stroke={stroke}/><path d="M 22 30 L 80 30" stroke={stroke} markerEnd="url(#arr)"/></svg>;
    case "altruism":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/>{([[30,20,"#e07b7b"],[42,30,"#7dd3a8"],[34,42,"#e0a86e"],[60,28,"#b08fe0"],[72,40,"#7dd3a8"],[64,18,"#e07b7b"]] as [number,number,string][]).map(([x,y,c],i)=>(<circle key={i} cx={x} cy={y} r="2.5" fill={c}/>))}</svg>;
    case "pairs":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/>{[[24,20],[28,22],[60,30],[64,32],[40,46],[44,44]].map((p,i)=>(<circle key={i} cx={p[0]} cy={p[1]} r="2" fill={stroke}/>))}</svg>;
    case "stripe":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/><rect x="44" y="4" width="12" height="52" fill={fill} stroke={stroke}/></svg>;
    case "diamond":
      return <svg viewBox="0 0 100 60">{[[25,18],[75,18],[25,42],[75,42]].map((p,i)=>(<polygon key={i} points={`${p[0]},${p[1]-7} ${p[0]+7},${p[1]} ${p[0]},${p[1]+7} ${p[0]-7},${p[1]}`} fill={fill} stroke={stroke}/>))}</svg>;
    case "wall":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/><line x1="30" y1="6" x2="30" y2="40" stroke={dim} strokeWidth="2"/><line x1="50" y1="20" x2="50" y2="56" stroke={dim} strokeWidth="2"/><line x1="70" y1="6" x2="70" y2="40" stroke={dim} strokeWidth="2"/><circle cx="86" cy="30" r="3" fill={stroke}/></svg>;
    case "density":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/><circle cx="50" cy="30" r="14" fill="none" stroke={stroke} strokeDasharray="2 2"/>{[[44,28],[56,30],[48,36],[52,24],[58,36],[42,32]].map((p,i)=>(<circle key={i} cx={p[0]} cy={p[1]} r="1.8" fill={stroke}/>))}</svg>;
    case "split":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/><rect x="2" y="2" width="20" height="56" fill={fill} stroke={stroke}/><rect x="78" y="2" width="20" height="56" fill={fill} stroke={stroke}/></svg>;
    case "drift":
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/><circle cx="30" cy="30" r="10" fill="none" stroke={stroke} opacity={0.3} strokeDasharray="2 2"/><circle cx="50" cy="30" r="10" fill="none" stroke={stroke} opacity={0.55} strokeDasharray="2 2"/><circle cx="70" cy="30" r="10" fill={fill} stroke={stroke}/></svg>;
    case "free":
    default:
      return <svg viewBox="0 0 100 60"><rect x="2" y="2" width="96" height="56" fill="none" stroke={dim} strokeDasharray="2 3"/><text x="50" y="35" textAnchor="middle" fontFamily="JetBrains Mono" fontSize="10" fill={dim}>NO ART</text></svg>;
  }
}
