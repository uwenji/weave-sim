// Colour maps and palettes, shared by the view, the legend, the plots and the Rhino export.
// The scientific maps are matplotlib's (64 samples each, interpolated to 256); the result palettes follow the Python
// viewer (weavesim/app/theme.py): PALETTES maps a palette name to (signed map, unsigned map) exactly as theme.PALETTES.
// The lath families and strands use Paul Tol's schemes as in LocalSH.
// Every colour here is sRGB (0–1); the view converts to linear before it hands colours to WebGL.
const HEX64 = {
  viridis: '44015446075a470d6047136548186a481d6f482374482878472d7b46327e453781443b844240864045883e49893d4e8a3a538b38588c365c8d34608d32648e31688e2f6c8e2d708e2c738e2a778e297b8e277f8e26828e24868e238a8d218e8d20928c1f968b1f9a8a1f9e891fa18721a58523a98326ad812ab07f2fb47c35b7793bbb7542be714ac16d52c5695ac86465cb5e6ece5877d15381d34d8bd64695d840a0da39aadc32b5de2bc0df25cae11fd5e21adfe318eae51af4e61efde725',
  magma: '00000402010903031206051a0a08220e0b2b130d34180f3d1d114722115029115a2f116336106b3d0f71440f764a107952137c59157e5f187f651a806b1d81721f817822817e24828426818b2981912b81982d809e2f7fa5317eab337cb2357bba3878c03a76c73d73cd4071d3436ed9466bdf4a68e44f64e95462ed5a5ff1605df4675cf66e5cf8765cfa7d5efb8560fc8e64fd9668fe9d6cfea571feac76feb47bfebb81fec287feca8dfed194fed89afde0a1fde7a9fceeb0fcf6b8fcfdbf',
  plasma: '0d088719068c2206902a059331059738049a3f049c46039f4c02a15302a35901a56001a66600a76c00a87201a87801a88004a88606a68b0aa5910ea39613a19c179ea11b9ba62098ab2494b02991b42e8db83289bd3786c13b82c5407ec9447acd4a76d14e72d5536fd8576bdb5c68de6164e26561e56a5de76f5aea7457ed7953ef7e50f1834cf48849f68d45f79342f99a3efb9f3afca537fdab33fdb130feb72dfebd2afdc328fdca26fcd025fbd724f9dd25f7e425f5eb27f2f227f0f921',
  inferno: '00000402010a04031207051b0b072410092d150b371b0c41210c4a280b532f0a5b3609613d0965440a684a0c6b510e6c59106e5f136e65156e6c186e721a6e781c6d7f1e6c85216b8c23699225689827669f2a63a52c60ab2f5eb1325ab73557bf3952c43c4eca404acf4446d44842d94d3dde5238e25734e65d2fea632aed6925f06f20f3761bf57d15f78410f98b0bfa9407fb9b06fca309fcaa0ffcb216fbba1ffac228f9c932f7d13df5d949f4e156f2e865f1ef75f3f586f6fa96fcffa4',
  cividis: '00224e00255400285b002b62002e6a0030700533711235701a386f213b6e273e6e2d416d32436d36466c3b496c3f4c6c444f6c48526c4c556c50576c545a6d575d6d5b606e5e636f62656f656870696b716c6e727071737374757777767a7a787e7d788280798683798a86788e8978928c78958f779992779d9576a19975a59c74a99f73ada272b1a570b6a96fbaac6dbfb06bc3b369c7b767ccba64d0be62d4c15fd9c55cddc858e1cc55e6d051ead34cefd748f3db42f8df3cfde334fee838',
  RdBu: '05306109386d0e41791249841752901b5a9c1f63a8246aae2a71b22f79b53480b93a87bd3f8ec04695c4529dc85fa5cd6eaed27bb6d687beda93c6de9dcbe1a7d0e4b1d5e7bbdaeac5dfeccfe4efd5e7f1dbeaf2e1edf3e7f0f4edf2f5f3f5f6f8f4f2f9f0ebf9ebe3fae7dcfbe3d4fcdecdfdd9c4fbd0b9fac8aff8bfa4f7b799f6af8ef4a683f09c7beb9172e6866ae17860dc6e57d7634fd25849cc4c44c6413ec13639bb2a34b61f2eae172aa21328960f278a0b257f082373042167001f',
  PuOr: '2d004b330655390c5e3f12684518714c1f7b522584582e8c5f3a916646976d529c745ea27b6aa88275ad8a7eb39287b99b92c1a39bc7aba3cdb3acd2b9b3d6bfbbdac5c2decbc9e2d1d1e6d7d8eadcddece1e2eee5e7f0eaebf2eff0f4f4f4f6f8f5f1f9f1e6faeedcfbead2fce6c8fde3befedeb3fed8a6fed299fdcc8cfdc57ffdbf72fdb965f9b158f4a84cf0a03fea9530e68d23e18417db7d12d47610cd700ec6690cbe630ab75c07b05606a85206a04d079749078f44078740087f3b08',
  coolwarm: '3b4cc03f53c6445acc4961d24e68d8536edd5875e15d7ce66282ea6788ee6c8ff17295f4779af77da0f982a6fb88abfd8fb1fe94b6ff9abbff9fbfffa5c3feaac7fdafcafcb5cdfabad0f8bfd3f6c4d5f3c9d7f0cdd9ecd2dbe8d6dce4dadce0dfdbd9e3d9d3e7d7ceead4c8edd1c2f0cdbbf2cab5f4c6aff5c1a9f6bda2f7b89cf7b396f7ad90f7a889f6a283f59c7df39475f18d6fee8669ec7f63e9785de57058e26952de614dd95847d55042d0473dcb3e38c53334c0282fba162bb40426',
  blues: 'f7fbfff4f9fef1f7fdeef5fceaf3fbe7f1fae4eff9e1edf8deebf7dbe9f6d8e7f5d5e5f4d2e3f3cfe1f2ccdff1c9ddf0c4daeebfd8edbad6ebb5d4e9b0d2e7abd0e6a6cee4a1cbe29cc9e195c5df8fc2de89bedc82bbdb7cb7da75b4d86fb0d768acd563a8d35da5d158a1cf539ecd4e9acb4997c94493c73f8fc53b8bc23787c03383be2f7fbc2b7bba2676b82272b61e6db21b69af1865ac1561a9125da60e59a20b559f08519c084d9608499008458a084184083c7d08387708347108306b',
  turbo: '30123b351e58392a733d358b4040a2424bb54456c74661d6466be34776ee4680f6458afc4294ff3d9efe37a8fa2fb2f427bee920c7df1bd0d518d7ca18dec01ae4b620eaac2aefa135f39443f78752fa7a61fc6c71fe5f80ff538fff499cfe40a9fb39b4f836bef434c8ef34d2e935dbe236e3db38ebd339f1cb3af6c33afaba39fcb136fea732fe9b2dfe9029fc8423f9751df56918f15d13ec530fe7490ce14109da3907d23105ca2a04c12302b71d02ac1701a11201950d018808027a0403',
  reds: 'fff5f0fff2ecfff0e8ffede5feeae1fee8ddfee5d9fee3d6fee0d2fedbccfdd7c6fdd2bffdcdb9fdc9b3fcc4adfcbfa7fcb99ffcb499fcaf93fcaa8dfca588fca082fc9b7cfc9576fc9070fc8b6bfc8666fc8161fb7c5cfb7757fb7252fb6d4dfa6648f96044f75b40f6553cf44f39f34935f14331f03d2dec382be83429e32f27de2b25da2723d52221d11e1fcc191ec7171cc2161bbd151ab81419b31218af1117aa1016a50f159d0d14960b138e09128608117e061077040f6f020e67000d',
  twilight: 'e2d9e2ddd9e0d6d7ddcdd3d8c2ced4b6c8cfaac2cc9ebbc993b4c689adc580a5c3779dc27195c06b8dbf6785be647cbc6173ba606ab75f61b45f58b05e4eab5e43a55d3a9e5b309558278b551f7f4f19724915644212573c114b36114232123a3212373711393e113c4713405014445b164865194c701c4e7a20508425508e2c509633509e3b50a54350ab4b50b15452b65d54ba6657be705bc27a61c58468c78f72ca997ccca287ceac94d1b5a1d4beafd8c6bddccdcaded3d4e1d7dce2d9e2',
};

function buildLUT(hexs, n = 256) {
  const k = hexs.length / 6, A = [];
  for (let i = 0; i < k; i++) A.push([0, 2, 4].map(o => parseInt(hexs.substr(6 * i + o, 2), 16) / 255));
  const lut = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1) * (k - 1), j = Math.min(Math.floor(t), k - 2), f = t - j;
    for (let c = 0; c < 3; c++) lut[3 * i + c] = A[j][c] * (1 - f) + A[j + 1][c] * f;
  }
  return lut;
}
export const LUT = Object.fromEntries(Object.entries(HEX64).map(([k, v]) => [k, buildLUT(v)]));
LUT.BuRd = LUT.RdBu;                                               // older name (blue → red)

/** Python theme.PALETTES: name → [map for signed layers, map for unsigned layers] */
export const PALETTES = {
  default: ['coolwarm', 'viridis'], viridis: ['viridis', 'viridis'], magma: ['magma', 'magma'], plasma: ['plasma', 'plasma'],
  inferno: ['inferno', 'inferno'], cividis: ['cividis', 'cividis'], RdBu: ['RdBu', 'RdBu'], PuOr: ['PuOr', 'PuOr'],
  blues: ['coolwarm', 'blues'], turbo: ['turbo', 'turbo'],
};
export const PALETTE_NAMES = Object.keys(PALETTES);
export const cmapFor = (palette, signed) => (PALETTES[palette] || PALETTES.default)[signed ? 0 : 1];

/** one colour of a LUT at t ∈ [0, 1] → [r, g, b] (sRGB) */
export function lutColor(name, t) { const lut = LUT[name] || LUT.viridis; const k = Math.round(Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0)) * 255); return [lut[3 * k], lut[3 * k + 1], lut[3 * k + 2]]; }
/** CSS gradient of a LUT (legend bar); `bands` > 0 draws that many flat steps */
export function lutGradient(name, dir = 'to right', bands = 0) {
  const css = (t) => { const [r, g, b] = lutColor(name, t); return `rgb(${Math.round(255 * r)},${Math.round(255 * g)},${Math.round(255 * b)})`; };
  const stops = [];
  if (bands > 0) for (let i = 0; i < bands; i++) { const c = css((i + 0.5) / bands); stops.push(`${c} ${(100 * i / bands).toFixed(2)}%`, `${c} ${(100 * (i + 1) / bands).toFixed(2)}%`); }
  else for (let i = 0; i <= 16; i++) stops.push(`${css(i / 16)} ${(100 * i / 16).toFixed(2)}%`);
  return `linear-gradient(${dir}, ${stops.join(', ')})`;
}
/** 256 × 1 RGBA bytes of a LUT (sRGB), for a texture */
export function lutBytes(name) { const lut = LUT[name] || LUT.viridis, out = new Uint8Array(256 * 4); for (let i = 0; i < 256; i++) { for (let c = 0; c < 3; c++) out[4 * i + c] = Math.round(255 * lut[3 * i + c]); out[4 * i + 3] = 255; } return out; }

export const hex = (h) => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
export const toHex = (rgb) => '#' + rgb.map(v => Math.round(255 * Math.max(0, Math.min(1, v))).toString(16).padStart(2, '0')).join('');
/** sRGB → linear (what WebGL vertex / instance colours expect) */
export const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

// --- the Python viewer's colours (theme.py, viewer.py, joints.py, diagrams.py) ------------------------------------------
/** lath families (the LocalSH colours): Paul Tol's high-contrast scheme for the first three — deep blue, warm gold,
 *  muted crimson — then his light scheme (orange, light yellow, pink, light blue, light cyan, mint) for weaves with more
 *  families; grey for an unassigned strand (family < 0). The view lifts them 18 % toward white on the dark canvas. */
export const FAMILY_HEX = ['#004488', '#DDAA33', '#BB5566', '#EE8866', '#EEDD88', '#FFAABB', '#77AADD', '#99DDFF', '#44BB99'];
export const FAMILY_COLORS = FAMILY_HEX.map(hex);
export const UNASSIGNED = hex('#8C8C8C');
export const familyColor = (f) => (f >= 0 ? FAMILY_COLORS[f % FAMILY_COLORS.length] : UNASSIGNED);
/** lath groups (viewer.GROUP_COLORS) */
export const GROUP_COLORS = [[0.80, 0.66, 0.46], [0.30, 0.55, 0.75], [0.75, 0.40, 0.35], [0.45, 0.65, 0.40], [0.60, 0.45, 0.70]];
export const TIMBER = [0.80, 0.66, 0.46];                          // "pure colour" default
export const PLAIN_LATH = TIMBER;
export const OVER_COLOR = [0.82, 0.60, 0.36], UNDER_COLOR = [0.55, 0.62, 0.70];
export const PIECE_COLORS = [[0.86, 0.72, 0.50], [0.55, 0.62, 0.70]];
export const CUT_COLOR = [0.15, 0.15, 0.17];
export const KNOT = [0.25, 0.25, 0.27], KNOT_BOUNDARY = [0.62, 0.62, 0.62], KNOT_FLIP = [0.95, 0.60, 0.10], KNOT_CONFLICT = [0.85, 0.15, 0.55];
export const LOAD = [0.80, 0.15, 0.15], ANCHOR_FULL = [0.70, 0.10, 0.10], ANCHOR_PIN = [0.92, 0.55, 0.10], ANCHOR_PART = [0.92, 0.82, 0.25], PROBE = [0.10, 0.62, 0.80];
export const SELECTION = [0.98, 0.78, 0.10];
/** member diagrams (diagrams.py): positive / negative fill, outline, undeformed shape */
export const DIAG_POS = [0.30, 0.62, 0.40], DIAG_NEG = [0.82, 0.38, 0.32], DIAG_OUTLINE = [0.18, 0.20, 0.22], DIAG_UNDEFORMED = [0.70, 0.70, 0.70];
/** one colour per strand "by strand" (Paul Tol muted) */
export const STRAND_HEX = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499'];
export const STRAND_COLORS = STRAND_HEX.map(hex);
/** stick (0) / slip (1) and contact types FF / FE / EE */
export const STATE_HEX = ['#4477AA', '#EE6677'];
export const CTYPE_HEX = ['#4477AA', '#CCBB44', '#EE6677'];
