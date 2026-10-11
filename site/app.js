// The motion browser. One three.js scene holds two figures that can play a motion: the robot
// of the URDF, and a VRM character. The cards and the viewer each pose the figure on show and
// draw it in turn, so the page loads each model only once.
//
// scripts/build_site.py writes a collection of motions: a folder with index.json.gz, the
// robot and the packs. collections.json lists the collections that the page shows: data/
// beside the page, or folders on the Hugging Face Hub. A motion is a byte range of a pack,
// with float32 rows of [pelvis xyz, pelvis quaternion xyzw, then the angle of each joint
// that its index entry lists]. character.vrm beside the page is the character. Without it,
// the page shows only the robot.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

// The URDF and the motions are z-up, so the cameras are too.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const COLLECTIONS = "collections.json";
const CHARACTER = "character.vrm";
const SESSION_KEY = "lite-motion:session";
const FIGURE_KEY = "lite-motion:figure";
const PAGE_SIZE = 48;
const MAX_PIXEL_RATIO = 2;
const FULL_MOTIONS_KEPT = 8;

// Metres and radians. A camera aims at the mid-height of the Lite above its pelvis, from a
// three-quarter front view, so a card frames the whole robot as it moves.
const LOOK_HEIGHT = 0.64;
const CARD_DISTANCE = 3.2;
const VIEWER_DISTANCE = 3.8;
const AZIMUTH = 0.6;
const ELEVATION = 0.17;

// The stage is the same room in either theme. These match --stage and --gold in style.css.
const STAGE_COLOR = 0x12233a;
const GOLD = 0xe8a612;

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ── Data ─────────────────────────────────────────────────────────────────────

async function fetchOk(url, headers = {}) {
  const response = await fetch(url, { headers });
  if (!response.ok) throw new Error(`${url} answered HTTP ${response.status}`);
  return response;
}

const fetchJson = async (url, headers) => (await fetchOk(url, headers)).json();

// The Hub serves a large file as stored, so the index comes gzipped and the page unpacks it.
async function fetchGzipJson(url, headers) {
  const body = (await fetchOk(url, headers)).body.pipeThrough(new DecompressionStream("gzip"));
  return new Response(body).json();
}

// The Hub answers each read of a file with a redirect to its CDN. The page reads a pack at
// the CDN address of its last read, and goes back through the Hub when that address expires.
const packAddresses = new Map();  // pack URL -> CDN URL

async function fetchRange(url, offset, bytes, headers) {
  const range = `bytes=${offset}-${offset + bytes - 1}`;
  let response = null;
  if (packAddresses.has(url)) {
    response = await fetch(packAddresses.get(url), { headers: { Range: range } }).catch(() => null);
  }
  if (response?.status !== 206) response = await fetchOk(url, { ...headers, Range: range });
  if (response.status !== 206) {
    throw new Error(`${url} answered with the whole file, not a range. Serve the site with scripts/serve.py.`);
  }
  packAddresses.set(url, response.url);
  return new Float32Array(await response.arrayBuffer());
}

// Fetch the rows of `entry` at `[pack, offset, bytes]`: its motion or its preview.
function fetchRows(entry, [pack, offset, bytes]) {
  const collection = entry.collection;
  return fetchRange(`${collection.url}${collection.packs[pack]}`, offset, bytes, collection.headers);
}

// Load each collection that the visitor can read, and merge them into one index. A private
// collection needs a signed-in visitor whose account can read it.
async function loadCollections() {
  const listed = (await fetchJson(COLLECTIONS)).filter((collection) => !collection.private || session);
  const loaded = await Promise.allSettled(listed.map(async (collection) => {
    const headers = collection.private ? { Authorization: `Bearer ${session.accessToken}` } : {};
    const data = await fetchGzipJson(`${collection.url}index.json.gz`, headers);
    return { ...collection, headers, data };
  }));
  const collections = [];
  for (const [i, result] of loaded.entries()) {
    if (result.status === "fulfilled") collections.push(result.value);
    else console.error(`Collection ${listed[i].url} did not load:`, result.reason);
  }
  if (collections.length === 0) throw new Error(`No collection in ${COLLECTIONS} loaded`);

  const first = collections[0].data;
  const merged = {
    robot: first.robot, viewer: first.viewer, joint_names: first.joint_names,
    groups: first.groups ?? [], robots: first.robots ?? [], sources: {}, motions: [],
  };
  for (const { data, ...collection } of collections) {
    if (data.joint_names.join() !== merged.joint_names.join()) {
      console.error(`Collection ${collection.url} has other joints than ${collections[0].url}, so the page skips it`);
      continue;
    }
    collection.packs = data.packs;
    Object.assign(merged.sources, data.sources);
    for (const entry of data.motions) {
      entry.collection = collection;
      entry.joints = data.joint_sets[entry.joint_set];
      // An older index has no active_sets, so take the joints that move as the driven joints.
      entry.active = data.active_sets?.[entry.active_set] ?? entry.joints;
      merged.motions.push(entry);
    }
  }
  return { index: merged, robotFrom: collections[0] };
}

// ── Sign in ──────────────────────────────────────────────────────────────────

// A static Space with `hf_oauth: true` in its card defines the OAuth client of the sign-in.
// Elsewhere the page has no sign-in, and shows only the public collections.
const oauthClient = window.huggingface?.variables?.OAUTH_CLIENT_ID;
let session = null;  // { accessToken, accessTokenExpiresAt, userInfo } of a signed-in visitor

function readStorage(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;  // a sandboxed frame or blocked site data refuses storage
  }
}

function writeStorage(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch (error) {
    console.warn(error);
  }
}

// The session kept from an earlier visit, or the one the Hub just sent the visitor back with.
async function restoreSession() {
  if (!oauthClient) return null;
  let result;
  try {
    result = JSON.parse(readStorage(SESSION_KEY));
  } catch {
    result = null;
  }
  if (!result && new URLSearchParams(location.search).has("code")) {
    try {
      const { oauthHandleRedirectIfPresent } = await import("@huggingface/hub");
      result = await oauthHandleRedirectIfPresent();
    } catch (error) {
      console.error(error);
    }
    // The Hub accepts a code only once, so remove it before a reload sends it again.
    replaceUrl(location.pathname + location.hash);
    if (result) writeStorage(SESSION_KEY, JSON.stringify(result));
  }
  if (result && new Date(result.accessTokenExpiresAt) <= new Date()) {
    writeStorage(SESSION_KEY, null);
    result = null;
  }
  return result || null;
}

async function signIn() {
  const { oauthLoginUrl } = await import("@huggingface/hub");
  // The Hub sends the visitor back to this page, without the hash of the open motion.
  location.href = await oauthLoginUrl({
    redirectUrl: location.origin + location.pathname,
    scopes: window.huggingface.variables.OAUTH_SCOPES,
  });
}

function signOut() {
  writeStorage(SESSION_KEY, null);
  location.reload();
}

function showAccount() {
  if (!oauthClient) return;
  const account = elements.account;
  const button = document.createElement("button");
  button.type = "button";
  if (session) {
    const avatar = document.createElement("img");
    avatar.src = session.userInfo.picture;
    avatar.alt = "";
    const name = document.createElement("span");
    name.textContent = session.userInfo.preferred_username;
    button.textContent = "Sign out";
    button.addEventListener("click", signOut);
    account.replaceChildren(avatar, name, button);
  } else {
    button.textContent = "Sign in with Hugging Face";
    button.title = "Show the collections that your Hugging Face account can read";
    button.addEventListener("click", signIn);
    account.replaceChildren(button);
  }
  account.hidden = false;
}

// ── Robot ────────────────────────────────────────────────────────────────────

// A figure is a model that a clip can pose. The stage draws `object`. The clip puts `root` at
// the pelvis, and sets each of `joints`, by robot joint name, with `setJoint(joint, value)`.
// `posedBy` is the clip that posed the figure last. `update(clip, time)` applies the pose to
// what the stage draws.

// Build the link tree of robot.json, with the meshes of robot.bin. Each joint becomes two
// groups: one fixed at the joint origin, and one inside it that the joint moves.
function buildRobot(description, buffer) {
  const materials = new Map();
  const material = ([r, g, b]) => {
    const key = `${r} ${g} ${b}`;
    if (!materials.has(key)) {
      materials.set(key, new THREE.MeshStandardMaterial({
        color: new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace),
        roughness: 0.55,
        metalness: 0.05,
        // An STL has a normal per face, so the mesh has no vertex normals and is shaded flat.
        flatShading: true,
      }));
    }
    return materials.get(key);
  };

  const links = new Map();
  for (const link of description.links) {
    const group = new THREE.Group();
    group.name = link.name;
    for (const visual of link.visuals) {
      const geometry = new THREE.BufferGeometry();
      const vertices = new Float32Array(buffer, visual.offset, visual.vertices * 3);
      const triangles = new Uint32Array(buffer, visual.offset + vertices.byteLength, visual.triangles * 3);
      geometry.setAttribute("position", new THREE.BufferAttribute(vertices, 3));
      geometry.setIndex(new THREE.BufferAttribute(triangles, 1));
      const mesh = new THREE.Mesh(geometry, material(visual.color));
      mesh.castShadow = true;
      group.add(mesh);
    }
    links.set(link.name, group);
  }

  const joints = new Map();
  for (const joint of description.joints) {
    const origin = new THREE.Group();
    origin.position.fromArray(joint.xyz);
    origin.quaternion.fromArray(joint.quat);
    const moving = new THREE.Group();
    origin.add(moving);
    moving.add(links.get(joint.child));
    links.get(joint.parent).add(origin);
    if (joint.type !== "fixed") {
      joints.set(joint.name, { type: joint.type, group: moving, axis: new THREE.Vector3().fromArray(joint.axis) });
    }
  }
  const root = links.get(description.root);
  return { object: root, root, joints, posedBy: null, setJoint, update() {} };
}

function setJoint(joint, value) {
  if (joint.type === "prismatic") joint.group.position.copy(joint.axis).multiplyScalar(value);
  else joint.group.quaternion.setFromAxisAngle(joint.axis, value);
}

// ── Character ────────────────────────────────────────────────────────────────

// A VRM is y-up and faces +z, with its left along +x. The robot is z-up and faces +x, with
// its left along +y. VRM_TO_ROBOT turns the VRM frame into the robot frame.
const VRM_TO_ROBOT = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(
  new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(1, 0, 0),
));
const ROBOT_TO_VRM = VRM_TO_ROBOT.clone().invert();

// Seconds. If a motion plays on by up to this much, the spring bones of the hair and the
// clothes follow it. If it jumps further, or back, they start again from rest.
const SPRING_STEP = 0.1;

// The bones of the character, each with the robot joints that turn it, in the order that the
// URDF chains them. Where the robot stacks yaw, pitch and roll joints at one point, the
// character has one ball joint, and its bone takes all of them. The pelvis of the robot moves
// the hips.
function characterBones() {
  const bones = {
    chest: ["waist_yaw", "waist_roll", "waist_pitch"],
    head: ["neck_yaw", "neck_roll", "neck_pitch"],
  };
  for (const side of ["left", "right"]) {
    bones[`${side}UpperLeg`] = [`${side}_hip_pitch`, `${side}_hip_roll`, `${side}_hip_yaw`];
    bones[`${side}LowerLeg`] = [`${side}_knee_pitch`];
    bones[`${side}Foot`] = [`${side}_ankle_yaw`, `${side}_ankle_pitch`, `${side}_ankle_roll`];
    bones[`${side}UpperArm`] = [`${side}_shoulder_pitch`, `${side}_shoulder_roll`, `${side}_shoulder_yaw`];
    bones[`${side}LowerArm`] = [`${side}_elbow_pitch`];
    bones[`${side}Hand`] = [`${side}_wrist_yaw`, `${side}_wrist_roll`, `${side}_wrist_pitch`];
    // Robot finger 1 is the thumb. A finger turns two ways at its knuckle, j1 and j2, then
    // bends at j3 and j4.
    for (const [i, finger] of ["Thumb", "Index", "Middle", "Ring", "Little"].entries()) {
      const joint = (j) => `${side}_finger${i + 1}_j${j}`;
      const phalanges = finger === "Thumb" ? ["Metacarpal", "Proximal", "Distal"] : ["Proximal", "Intermediate", "Distal"];
      bones[`${side}${finger}${phalanges[0]}`] = [joint(1), joint(2)];
      bones[`${side}${finger}${phalanges[1]}`] = [joint(3)];
      bones[`${side}${finger}${phalanges[2]}`] = [joint(4)];
    }
  }
  return bones;
}

// The axis of each robot joint with the robot at rest, in the frame of the character. At rest
// both stand in a T-pose and face the same way, so a joint turns the bone of the character
// about the same axis as the robot.
function restAxes(description) {
  const byChild = new Map(description.joints.map((joint) => [joint.child, joint]));
  const rotations = new Map([[description.root, new THREE.Quaternion()]]);
  const rotation = (link) => {
    if (!rotations.has(link)) {
      const joint = byChild.get(link);
      rotations.set(link, rotation(joint.parent).clone().multiply(new THREE.Quaternion().fromArray(joint.quat)));
    }
    return rotations.get(link);
  };
  const axes = new Map();
  for (const joint of description.joints) {
    const axis = new THREE.Vector3().fromArray(joint.axis).applyQuaternion(rotation(joint.child));
    axes.set(joint.name, axis.applyQuaternion(ROBOT_TO_VRM));
  }
  return axes;
}

// Load the character, or return null if the page has none.
async function loadVrm(url) {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  try {
    const gltf = await loader.loadAsync(url);
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.combineSkeletons(gltf.scene);
    return gltf.userData.vrm;
  } catch (error) {
    console.warn(`${url} did not load, so the page shows only the robot:`, error);
    return null;
  }
}

// Build the character as a figure that takes the joint angles of the robot. Its bones turn on
// the normalized rig of the VRM, where each bone at rest has no rotation in the frame of the
// model. `sole` is the height of the lowest point of the robot at rest, relative to its
// pelvis. The character puts its hip joints at about the same point as the robot, and its
// soles at the same height.
function buildCharacter(vrm, description, sole) {
  const humanoid = vrm.humanoid;
  const axes = restAxes(description);
  const joints = new Map();
  const bones = [];
  for (const [name, chain] of Object.entries(characterBones())) {
    const node = humanoid.getNormalizedBoneNode(name);
    if (!node) continue;  // a VRM may leave out the fingers
    const turns = chain.filter((joint) => axes.has(joint)).map((joint) => {
      if (!joints.has(joint)) joints.set(joint, { angle: 0 });
      return { joint: joints.get(joint), axis: axes.get(joint) };
    });
    bones.push({ node, turns });
  }

  const hips = humanoid.getNormalizedBoneNode("hips");
  const legs = ["leftUpperLeg", "rightUpperLeg"].map((name) => humanoid.getNormalizedBoneNode(name).position);
  const legsFromHips = legs[0].clone().add(legs[1]).multiplyScalar(0.5).applyQuaternion(VRM_TO_ROBOT);
  const hipJoints = description.joints.filter((joint) => joint.name.endsWith("_hip_pitch"));
  const offset = new THREE.Vector3(
    (hipJoints[0].xyz[0] + hipJoints[1].xyz[0]) / 2 - legsFromHips.x,
    (hipJoints[0].xyz[1] + hipJoints[1].xyz[1]) / 2 - legsFromHips.y,
    sole + hips.position.y,  // a VRM stands with its soles at its origin
  );

  const object = new THREE.Group();
  object.quaternion.copy(VRM_TO_ROBOT);
  object.add(vrm.scene);
  vrm.scene.traverse((node) => {
    node.frustumCulled = false;  // a skinned mesh keeps the bounds of its rest pose
    if (node.isMesh) node.castShadow = true;
  });
  // Gravity pulls a spring bone along a world direction: -y for a VRM, -z on this stage.
  for (const joint of vrm.springBoneManager?.joints ?? []) joint.settings.gravityDir.applyQuaternion(VRM_TO_ROBOT);

  const root = new THREE.Object3D();
  const turn = new THREE.Quaternion();
  const last = { clip: null, time: 0 };
  function update(clip, time) {
    for (const { node, turns } of bones) {
      node.quaternion.identity();
      for (const { joint, axis } of turns) node.quaternion.multiply(turn.setFromAxisAngle(axis, joint.angle));
    }
    hips.quaternion.copy(ROBOT_TO_VRM).multiply(root.quaternion).multiply(VRM_TO_ROBOT);
    hips.position.copy(offset).applyQuaternion(root.quaternion).add(root.position).applyQuaternion(ROBOT_TO_VRM);

    const step = clip === last.clip ? time - last.time : -1;
    last.clip = clip;
    last.time = time;
    if (step >= 0 && step <= SPRING_STEP) {
      vrm.update(step);
    } else {
      vrm.update(0);
      object.updateMatrixWorld(true);
      vrm.springBoneManager?.reset();
    }
  }
  return { object, root, joints, posedBy: null, setJoint: (joint, value) => { joint.angle = value; }, update };
}

// ── Motion ───────────────────────────────────────────────────────────────────

const mix = (a, b, t) => a + (b - a) * t;
const quatA = new THREE.Quaternion();
const quatB = new THREE.Quaternion();

class Clip {
  constructor(rows, entry, jointNames) {
    this.rows = rows;
    this.fps = entry.fps;
    this.stride = 7 + entry.joints.length;
    this.frames = rows.length / this.stride;
    this.duration = (this.frames - 1) / this.fps;
    this.names = entry.joints.map((i) => jointNames[i]);
    this.targets = new Map();  // figure -> its joint for each joint of the clip
  }

  // Pose `figure` at `time` seconds, interpolated between the two frames around it.
  pose(figure, time) {
    const f = THREE.MathUtils.clamp(time * this.fps, 0, this.frames - 1);
    const i = Math.floor(f);
    const a = f - i;
    const r = this.rows;
    const p = i * this.stride;
    const q = Math.min(i + 1, this.frames - 1) * this.stride;

    // The clip skips a joint that the figure does not have. A joint that the clip does not
    // list is zero, but the clip that posed the figure last can have moved it, so a new clip
    // sets each joint to zero first.
    if (!this.targets.has(figure)) this.targets.set(figure, this.names.map((name) => figure.joints.get(name) ?? null));
    const targets = this.targets.get(figure);
    if (figure.posedBy !== this) {
      for (const joint of figure.joints.values()) figure.setJoint(joint, 0);
      figure.posedBy = this;
    }
    figure.root.position.set(mix(r[p], r[q], a), mix(r[p + 1], r[q + 1], a), mix(r[p + 2], r[q + 2], a));
    quatA.set(r[p + 3], r[p + 4], r[p + 5], r[p + 6]);
    quatB.set(r[q + 3], r[q + 4], r[q + 5], r[q + 6]);
    figure.root.quaternion.slerpQuaternions(quatA, quatB, a);
    for (let k = 0; k < targets.length; k++) {
      if (targets[k]) figure.setJoint(targets[k], mix(r[p + 7 + k], r[q + 7 + k], a));
    }
    figure.update(this, time);
  }

  // The direction the pelvis faces at `time`, in radians about z from the world x axis.
  heading(time) {
    const p = Math.round(THREE.MathUtils.clamp(time * this.fps, 0, this.frames - 1)) * this.stride;
    const [x, y, z, w] = this.rows.subarray(p + 3, p + 7);
    return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  }

  // The pelvis path over the floor, as a line a few millimetres above it.
  track() {
    const step = Math.max(1, Math.floor(this.frames / 4000));
    const points = [];
    for (let f = 0; f < this.frames; f += step) {
      points.push(this.rows[f * this.stride], this.rows[f * this.stride + 1], 0.004);
    }
    return new THREE.Float32BufferAttribute(points, 3);
  }
}

// ── Stage ────────────────────────────────────────────────────────────────────

function createStage(figures) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(STAGE_COLOR);
  scene.fog = new THREE.Fog(STAGE_COLOR, 6, 17);
  scene.add(new THREE.HemisphereLight(0xdde8ff, 0x2a3442, 1.5));

  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  Object.assign(key.shadow.camera, { left: -1.8, right: 1.8, top: 1.8, bottom: -1.8, near: 0.5, far: 12 });
  key.shadow.normalBias = 0.02;
  scene.add(key, key.target);

  const grid = floorGrid(30, 0.5, 0x33527c);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.ShadowMaterial({ opacity: 0.34 }));
  floor.receiveShadow = true;
  const trail = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: GOLD, transparent: true, opacity: 0.8 }),
  );
  scene.add(grid, floor, trail, ...figures.map((figure) => figure.object));

  // Keep the floor, the grid and the shadow of the key light around the figure, so a motion
  // can cross any distance. The grid moves in whole metres, so its lines stay in place.
  function center(x, y) {
    grid.position.set(Math.round(x), Math.round(y), 0.001);
    floor.position.set(x, y, 0);
    key.position.set(x + 1.8, y - 2.4, 5);
    key.target.position.set(x, y, 0);
  }
  return { scene, trail, center };
}

// A grid of `step` cells over a `size` square of the floor, as one segment per cell edge.
// A line that spans the whole floor runs behind the camera. Some renderers clip such a line
// incorrectly, and the floor under the robot shows no grid.
function floorGrid(size, step, color) {
  const cells = Math.round(size / step);
  const points = [];
  for (let i = 0; i <= cells; i++) {
    const a = i * step - size / 2;
    for (let j = 0; j < cells; j++) {
      const b = j * step - size / 2;
      points.push(a, b, 0, a, b + step, 0, b, a, 0, b + step, a, 0);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color }));
}

function aim(camera, x, y, azimuth, distance, elevation) {
  const across = distance * Math.cos(elevation);
  camera.position.set(
    x + across * Math.cos(azimuth),
    y + across * Math.sin(azimuth),
    LOOK_HEIGHT + distance * Math.sin(elevation),
  );
  camera.lookAt(x, y, LOOK_HEIGHT);
}

function createRenderer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  return renderer;
}

// ── Page state ───────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);
const elements = {
  search: $("search"), sources: $("sources"), drives: $("drives"), categories: $("categories"), count: $("count"),
  sort: $("sort"), grid: $("grid"), more: $("more"), notice: $("notice"),
  viewer: $("viewer"), canvas: $("viewer-canvas"), loading: $("viewer-loading"),
  play: $("play"), scrub: $("scrub"), clock: $("clock"), speed: $("speed"),
  follow: $("follow"), trail: $("trail"), prev: $("prev"), next: $("next"), close: $("close"),
  source: $("viewer-source"), name: $("viewer-name"), stats: $("viewer-stats"),
  openRerun: $("open-rerun"), command: $("command"), copy: $("copy"), account: $("account"),
};

let index = null;
const figures = {};  // "character" and "robot", each a figure
let figure = null;   // the one on show
let stage = null;
const byName = new Map();  // motion name -> index entry
// `categoryKey` is the picked category of one source, as "<source>:<category>", or null for
// all. `drives` holds the joint groups that a motion must drive in full.
const state = { query: [], sources: new Set(), drives: new Set(), categoryKey: null, sort: "name" };
let filtered = [];
let shown = 0;

// ── Cards ────────────────────────────────────────────────────────────────────

const cardRenderer = createRenderer(document.createElement("canvas"));
const cardCamera = new THREE.PerspectiveCamera(30, 4 / 3, 0.1, 60);
const cards = new Map();        // motion name -> card, kept across filters
const cardOf = new WeakMap();   // card element -> card
const posterQueue = new Set();  // cards to draw at rest
let hovered = null;

const cardObserver = new IntersectionObserver((records) => {
  for (const record of records) {
    const card = cardOf.get(record.target);
    card.visible = record.isIntersecting;
    if (card.visible) {
      loadPreview(card);
      if (!card.drawn) posterQueue.add(card);
    } else {
      release(card);
    }
  }
}, { rootMargin: "400px 0px" });

const moreObserver = new IntersectionObserver((records) => {
  if (records[0].isIntersecting) appendPage();
}, { rootMargin: "800px 0px" });

function createCard(entry) {
  const element = document.createElement("button");
  element.type = "button";
  element.className = "card";
  element.innerHTML = `
    <span class="tile"><canvas></canvas><span class="tag"></span><span class="length"></span></span>
    <span class="label"><span class="name"></span><span class="meta"></span></span>`;
  element.querySelector(".tag").textContent = index.sources[entry.source].title;
  element.querySelector(".length").textContent = formatLength(entry.seconds);
  element.querySelector(".name").textContent = cardTitle(entry);
  element.querySelector(".meta").textContent = entry.category;
  element.title = entry.name;

  const canvas = element.querySelector("canvas");
  const card = {
    entry, element, canvas, context: canvas.getContext("2d"),
    clip: null, loading: false, visible: false, drawn: false,
    time: 0, azimuth: 0, focus: new THREE.Vector2(),
  };
  element.addEventListener("pointerenter", (event) => { if (event.pointerType === "mouse") startHover(card); });
  element.addEventListener("pointerleave", () => stopHover(card));
  element.addEventListener("focus", () => startHover(card));
  element.addEventListener("blur", () => stopHover(card));
  element.addEventListener("click", () => openViewer(entry));
  cardOf.set(element, card);
  return card;
}

async function loadPreview(card) {
  if (card.clip || card.loading) return;
  card.loading = true;
  try {
    // A motion no longer than a preview is its own preview.
    const rows = await fetchRows(card.entry, card.entry.preview ?? card.entry.motion);
    card.clip = new Clip(rows, card.entry, index.joint_names);
    // A card rests on the middle of its preview, and plays on from there.
    card.time = card.clip.duration / 2;
    card.azimuth = card.clip.heading(card.time) + AZIMUTH;
    if (card.visible) posterQueue.add(card);
  } catch (error) {
    console.error(error);
  } finally {
    card.loading = false;
  }
}

// Free the pixels of a card far off screen. The card draws again when it comes back.
function release(card) {
  posterQueue.delete(card);
  card.canvas.width = 0;
  card.canvas.height = 0;
  card.drawn = false;
}

// Pose the figure as the clip of the card has it, draw it, and copy the frame into the card.
// `follow` is the fraction of the distance to the pelvis that the camera moves: 1 moves all
// the way.
function drawCard(card, follow) {
  const width = card.canvas.clientWidth;
  const height = card.canvas.clientHeight;
  if (!width || !height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO);
  const w = Math.round(width * ratio);
  const h = Math.round(height * ratio);
  if (card.canvas.width !== w || card.canvas.height !== h) {
    card.canvas.width = w;
    card.canvas.height = h;
  }
  const size = cardRenderer.getSize(new THREE.Vector2());
  if (size.x !== w || size.y !== h) cardRenderer.setSize(w, h, false);

  card.clip.pose(figure, card.time);
  const { x, y } = figure.root.position;
  card.focus.x = mix(card.focus.x, x, follow);
  card.focus.y = mix(card.focus.y, y, follow);
  stage.center(x, y);
  stage.trail.visible = false;
  cardCamera.aspect = w / h;
  cardCamera.updateProjectionMatrix();
  aim(cardCamera, card.focus.x, card.focus.y, card.azimuth, CARD_DISTANCE, ELEVATION);
  cardRenderer.render(stage.scene, cardCamera);
  card.context.drawImage(cardRenderer.domElement, 0, 0);
  card.drawn = true;
}

// Draw the cards that wait for a still frame, for about 10 ms per frame of the page.
function drawPosters() {
  const start = performance.now();
  for (const card of posterQueue) {
    if (performance.now() - start > 10) break;
    posterQueue.delete(card);
    if (card.visible && card.clip && card !== hovered) drawCard(card, 1);
  }
}

// Queue a new still frame for each visible card.
function redrawCards() {
  for (const card of cards.values()) {
    if (card.visible) posterQueue.add(card);
  }
}

function startHover(card) {
  if (hovered && hovered !== card) stopHover(hovered);
  hovered = card;
  card.element.classList.add("playing");
}

function stopHover(card) {
  if (hovered !== card) return;
  hovered = null;
  card.element.classList.remove("playing");
  if (card.clip) {
    card.time = card.clip.duration / 2;
    posterQueue.add(card);
  }
}

function playHovered(dt) {
  const card = hovered;
  if (!card?.clip || !card.visible) return;
  card.time += dt;
  let follow = 1 - Math.exp(-dt * 4);
  if (card.time > card.clip.duration) {
    card.time = 0;
    follow = 1;
  }
  drawCard(card, follow);
}

// ── Gallery ──────────────────────────────────────────────────────────────────

const SORTS = {
  name: (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }),
  longest: (a, b) => b.seconds - a.seconds,
  shortest: (a, b) => a.seconds - b.seconds,
  travel: (a, b) => b.travel - a.travel,
};

function cardTitle(entry) {
  return (entry.clip ?? entry.name.slice(entry.source.length + 1)).split("/").pop();
}

function formatLength(seconds) {
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(Math.floor(seconds - minutes * 60)).padStart(2, "0")}`;
}

function formatTotal(seconds) {
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)} h`;
  if (seconds >= 60) return `${Math.round(seconds / 60)} min`;
  return `${Math.round(seconds)} s`;
}

// A row of the filter rail. A `check` row toggles one of several options. Another row picks
// one option.
function filterRow(label, tally, pressed, check, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = check ? "filter-row check" : "filter-row";
  button.setAttribute("aria-pressed", String(pressed));
  button.innerHTML =
    `<span class="box" aria-hidden="true"></span><span class="label"></span><span class="tally"></span>`;
  button.querySelector(".label").textContent = label;
  button.querySelector(".tally").textContent = tally.toLocaleString();
  button.addEventListener("click", onClick);
  return button;
}

// Add `key` to `set` if it is not there, or else remove it, and apply the filters again.
function toggle(set, key) {
  if (!set.delete(key)) set.add(key);
  applyFilters();
}

// Apply the filters: redraw the source, drive and category lists, and start the grid again.
function applyFilters() {
  const searched = index.motions.filter((entry) => state.query.every((word) => entry.haystack.includes(word)));
  elements.sources.replaceChildren(...Object.entries(index.sources).map(([key, source]) => {
    const tally = searched.filter((entry) => entry.source === key).length;
    return filterRow(source.title, tally, state.sources.has(key), true, () => toggle(state.sources, key));
  }));

  const picked = searched.filter((entry) => state.sources.has(entry.source));
  elements.drives.replaceChildren(...index.groups.map(({ title }) => {
    const tally = picked.filter((entry) => entry.drives.has(title)).length;
    return filterRow(title, tally, state.drives.has(title), true, () => toggle(state.drives, title));
  }));

  // Each source has its own categories, so the list shows them by source.
  const matches = picked.filter((entry) => [...state.drives].every((title) => entry.drives.has(title)));
  const tallies = new Map();  // category key -> motions
  for (const entry of matches) tallies.set(entry.categoryKey, (tallies.get(entry.categoryKey) ?? 0) + 1);
  if (state.categoryKey && !tallies.has(state.categoryKey)) state.categoryKey = null;
  const pick = (key) => () => {
    state.categoryKey = state.categoryKey === key ? null : key;
    applyFilters();
  };
  const rows = [filterRow("All", matches.length, state.categoryKey === null, false, pick(null))];
  for (const [source, { title }] of Object.entries(index.sources)) {
    const keys = [...tallies.keys()].filter((key) => key.startsWith(`${source}:`)).sort();
    if (keys.length === 0) continue;
    const heading = document.createElement("p");
    heading.className = "group-label";
    heading.textContent = title;
    rows.push(heading, ...keys.map((key) =>
      filterRow(key.slice(source.length + 1), tallies.get(key), state.categoryKey === key, false, pick(key))));
  }
  elements.categories.replaceChildren(...rows);

  filtered = matches
    .filter((entry) => !state.categoryKey || entry.categoryKey === state.categoryKey)
    .sort(SORTS[state.sort]);
  const seconds = filtered.reduce((sum, entry) => sum + entry.seconds, 0);
  elements.count.innerHTML =
    `<strong>${filtered.length.toLocaleString()}</strong> motions · ${formatTotal(seconds)} of motion`;

  for (const element of elements.grid.children) {
    const card = cardOf.get(element);
    if (card === hovered) stopHover(card);
    cardObserver.unobserve(element);
    card.visible = false;
    release(card);
  }
  elements.grid.replaceChildren();
  shown = 0;
  if (filtered.length === 0) {
    showNotice("No motion matches", "Clear the search, or change the filters.");
  } else {
    elements.notice.hidden = true;
  }
  appendPage();
}

function appendPage() {
  const page = filtered.slice(shown, shown + PAGE_SIZE);
  for (const entry of page) {
    if (!cards.has(entry.name)) cards.set(entry.name, createCard(entry));
    const card = cards.get(entry.name);
    elements.grid.append(card.element);
    cardObserver.observe(card.element);
  }
  shown += page.length;
  // A new observation reports the sentinel at once. Thus if the sentinel stays in view after
  // a page, the next page loads too.
  moreObserver.unobserve(elements.more);
  if (shown < filtered.length) moreObserver.observe(elements.more);
}

function showNotice(title, detail) {
  const heading = document.createElement("h2");
  heading.textContent = title;
  const text = document.createElement("p");
  text.innerHTML = detail;
  elements.notice.replaceChildren(heading, text);
  elements.notice.hidden = false;
}

// ── Viewer ───────────────────────────────────────────────────────────────────

const viewer = {
  renderer: createRenderer(elements.canvas),
  camera: new THREE.PerspectiveCamera(35, 1, 0.05, 80),
  controls: null,
  entry: null,
  clip: null,
  time: 0,
  playing: true,
  scrubbing: false,
  request: 0,
  returnFocus: null,
};
viewer.controls = new OrbitControls(viewer.camera, elements.canvas);
viewer.controls.enableDamping = true;
viewer.controls.minDistance = 0.8;
viewer.controls.maxDistance = 14;
viewer.controls.maxPolarAngle = THREE.MathUtils.degToRad(88);

const motionCache = new Map();  // name -> promise of rows, the most recent last

function loadMotion(entry) {
  if (!motionCache.has(entry.name)) {
    motionCache.set(entry.name, fetchRows(entry, entry.motion));
    if (motionCache.size > FULL_MOTIONS_KEPT) motionCache.delete(motionCache.keys().next().value);
  }
  return motionCache.get(entry.name);
}

async function openViewer(entry) {
  if (hovered) stopHover(hovered);
  const request = ++viewer.request;
  if (elements.viewer.hidden) {
    viewer.returnFocus = document.activeElement;
    elements.viewer.hidden = false;
    document.body.classList.add("locked");
    elements.close.focus();
  }
  viewer.entry = entry;
  viewer.clip = null;
  describe(entry);
  replaceUrl(`#${entry.name}`);
  elements.loading.textContent = "Loading motion";
  elements.loading.hidden = false;

  let rows;
  try {
    rows = await loadMotion(entry);
  } catch (error) {
    motionCache.delete(entry.name);
    if (request === viewer.request) elements.loading.textContent = `This motion did not load: ${error.message}`;
    return;
  }
  if (request !== viewer.request) return;  // the visitor opened another motion meanwhile

  const clip = new Clip(rows, entry, index.joint_names);
  viewer.clip = clip;
  viewer.time = 0;
  elements.scrub.max = String(clip.frames - 1);
  clip.pose(figure, 0);
  const { x, y } = figure.root.position;
  aim(viewer.camera, x, y, clip.heading(0) + AZIMUTH, VIEWER_DISTANCE, ELEVATION + 0.05);
  viewer.controls.target.set(x, y, LOOK_HEIGHT);
  viewer.controls.update();
  stage.trail.geometry.dispose();
  stage.trail.geometry = new THREE.BufferGeometry().setAttribute("position", clip.track());
  setPlaying(!reducedMotion);
  elements.loading.hidden = true;
}

// Replace the URL without a reload. The hash names the open motion, so a link opens it. A
// sandboxed frame can refuse the change.
function replaceUrl(url) {
  try {
    history.replaceState(null, "", url);
  } catch (error) {
    console.warn(error);
  }
}

function closeViewer() {
  viewer.request++;
  viewer.clip = null;
  elements.viewer.hidden = true;
  document.body.classList.remove("locked");
  replaceUrl(location.pathname + location.search);
  viewer.returnFocus?.focus?.();
}

// Previous and Next walk the filtered grid, or every motion if the open motion is not in it.
function neighbour(step) {
  const list = filtered.includes(viewer.entry) ? filtered : index.motions;
  return list[list.indexOf(viewer.entry) + step] ?? null;
}

function go(step) {
  const entry = neighbour(step);
  if (entry) openViewer(entry);
}

function sourceRate(entry) {
  if (!entry.source_fps) return "Not in the store";
  if (entry.source_fps === entry.fps) return `${entry.fps} Hz, as recorded`;
  return `${entry.source_fps} Hz, resampled to ${entry.fps} Hz`;
}

// The joint groups that `entry` drives, in full or in part, and the groups that hold zero.
function describeDrives(entry) {
  const active = new Set(entry.active);
  const driven = [];
  const held = [];
  for (const { title, joints } of index.groups) {
    const count = joints.filter((joint) => active.has(joint)).length;
    if (count === joints.length) driven.push(title.toLowerCase());
    else if (count > 0) driven.push(`${title.toLowerCase()} (${count} of ${joints.length})`);
    else held.push(title.toLowerCase());
  }
  const list = (words) => words.length > 1 ? `${words.slice(0, -1).join(", ")} and ${words.at(-1)}` : words[0] ?? "none";
  const total = `${entry.active.length} of ${index.joint_names.length} joints`;
  const rest = held.length ? `. ${list(held).replace(/^./, (c) => c.toUpperCase())} hold zero.` : ".";
  return `${total}: ${list(driven)}${rest}`;
}

// The number of joints of each Lite robot that `entry` drives.
function describeRobots(entry) {
  const active = new Set(entry.active);
  return index.robots
    .map(({ title, joints }) => `${title} ${joints.filter((joint) => active.has(joint)).length}/${joints.length}`)
    .join(" · ");
}

function describe(entry) {
  const source = index.sources[entry.source];
  elements.source.textContent = `${source.title} · ${entry.category}`;
  elements.name.textContent = entry.name;

  const link = document.createElement("a");
  link.href = source.url;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = source.url.includes("huggingface.co") ? "Source dataset on the Hub" : "Source dataset";
  const rows = [
    ["Length", `${formatLength(entry.seconds)} · ${entry.frames.toLocaleString()} frames at ${entry.fps} Hz`],
    ["Source rate", sourceRate(entry)],
    ["Travel", `${entry.travel.toFixed(1)} m of pelvis path over the floor`],
    ["Skeleton", source.skeleton ?? "Unitree G1, 29 joints"],
    ["Drives", describeDrives(entry)],
    ["Robots", describeRobots(entry)],
    ["Clip", entry.clip ?? "Not in the local download of the source", "mono"],
    ["License", source.license],
    ["Upstream", link],
  ];
  elements.stats.replaceChildren(...rows.flatMap(([term, value, style]) => {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.append(value);
    if (style) dd.className = style;
    return [dt, dd];
  }));

  elements.openRerun.hidden = !source.hub;
  if (source.hub) elements.openRerun.href = index.viewer + encodeURIComponent(`${source.hub}${entry.name}.rrd`);
  elements.command.textContent = `uv run rerun dataset/*/${entry.name}.rrd ` +
    `--asset dataset/assets/${index.robot}.rrd dataset/blueprints/${index.robot}.rbl`;
  elements.prev.disabled = !neighbour(-1);
  elements.next.disabled = !neighbour(1);
}

function setPlaying(playing) {
  viewer.playing = playing;
  elements.play.classList.toggle("paused", !playing);
  elements.play.setAttribute("aria-label", playing ? "Pause" : "Play");
}

function stepFrame(step) {
  const clip = viewer.clip;
  if (!clip) return;
  setPlaying(false);
  const frame = THREE.MathUtils.clamp(Math.round(viewer.time * clip.fps) + step, 0, clip.frames - 1);
  viewer.time = frame / clip.fps;
}

function resizeViewer() {
  const width = elements.canvas.clientWidth;
  const height = elements.canvas.clientHeight;
  if (!width || !height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO);
  // setSize rounds the canvas size down, so compare with the rounded-down size.
  if (elements.canvas.width !== Math.floor(width * ratio) || elements.canvas.height !== Math.floor(height * ratio)) {
    viewer.renderer.setPixelRatio(ratio);
    viewer.renderer.setSize(width, height, false);
    viewer.camera.aspect = width / height;
    viewer.camera.updateProjectionMatrix();
  }
}

function tickViewer(dt) {
  const clip = viewer.clip;
  if (!clip) return;
  resizeViewer();
  let follow = 1 - Math.exp(-dt * 6);
  if (viewer.playing && !viewer.scrubbing) {
    viewer.time += dt * Number(elements.speed.value);
    if (viewer.time > clip.duration) {
      viewer.time = 0;
      follow = 1;
    }
  }
  clip.pose(figure, viewer.time);

  // Follow moves the camera and its target together, so the view angle that the visitor
  // picked stays the same while the figure crosses the floor.
  const { x, y } = figure.root.position;
  if (elements.follow.checked) {
    const target = viewer.controls.target;
    const dx = (x - target.x) * follow;
    const dy = (y - target.y) * follow;
    target.x += dx;
    target.y += dy;
    viewer.camera.position.x += dx;
    viewer.camera.position.y += dy;
  }
  viewer.controls.update();
  stage.center(x, y);
  stage.trail.visible = elements.trail.checked;
  viewer.renderer.render(stage.scene, viewer.camera);

  const frame = Math.round(viewer.time * clip.fps);
  if (!viewer.scrubbing) elements.scrub.value = String(frame);
  elements.clock.innerHTML =
    `${viewer.time.toFixed(2)} s<span class="frame"> · frame ${frame} / ${clip.frames - 1}</span>`;
}

elements.play.addEventListener("click", () => setPlaying(!viewer.playing));
elements.scrub.addEventListener("pointerdown", () => { viewer.scrubbing = true; });
window.addEventListener("pointerup", () => { viewer.scrubbing = false; });
elements.scrub.addEventListener("input", () => {
  if (viewer.clip) viewer.time = Number(elements.scrub.value) / viewer.clip.fps;
});
elements.prev.addEventListener("click", () => go(-1));
elements.next.addEventListener("click", () => go(1));
elements.close.addEventListener("click", closeViewer);
elements.copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(elements.command.textContent);
    elements.copy.textContent = "Copied";
  } catch {
    getSelection().selectAllChildren(elements.command);
    elements.copy.textContent = "Selected";
  }
  setTimeout(() => { elements.copy.textContent = "Copy"; }, 1600);
});

document.addEventListener("keydown", (event) => {
  if (elements.viewer.hidden) return;
  if (event.key === "Escape") {
    closeViewer();
    return;
  }
  const target = event.target;
  if (target.tagName === "SELECT" || (target.tagName === "INPUT" && target.type !== "checkbox")) return;
  const actions = {
    " ": () => setPlaying(!viewer.playing),
    ArrowLeft: () => stepFrame(-1),
    ArrowRight: () => stepFrame(1),
    ArrowUp: () => go(-1),
    ArrowDown: () => go(1),
  };
  if (!actions[event.key]) return;
  event.preventDefault();
  actions[event.key]();
});

function openFromHash() {
  const entry = byName.get(decodeURIComponent(location.hash.slice(1)));
  if (entry) openViewer(entry);
  else if (!elements.viewer.hidden) closeViewer();
}

// Show the figure `name` on the stage, in the viewer and on every card.
function showFigure(name) {
  figure = figures[name];
  for (const each of Object.values(figures)) each.object.visible = each === figure;
  for (const button of document.querySelectorAll("[data-figure]")) {
    button.setAttribute("aria-pressed", String(button.dataset.figure === name));
  }
  writeStorage(FIGURE_KEY, name);
  redrawCards();
}

// ── Start ────────────────────────────────────────────────────────────────────

let last = performance.now();
function loop(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (!elements.viewer.hidden) {
    tickViewer(dt);
  } else {
    playHovered(dt);
    drawPosters();
  }
  requestAnimationFrame(loop);
}

async function main() {
  session = await restoreSession();
  showAccount();
  try {
    const loaded = await loadCollections();
    index = loaded.index;
    const { url, headers } = loaded.robotFrom;
    const [description, buffer, vrm] = await Promise.all([
      fetchJson(`${url}robot.json`, headers),
      fetchOk(`${url}robot.bin`, headers).then((response) => response.arrayBuffer()),
      loadVrm(CHARACTER),
    ]);
    figures.robot = buildRobot(description, buffer);
    if (vrm) {
      const sole = new THREE.Box3().setFromObject(figures.robot.root).min.z;
      figures.character = buildCharacter(vrm, description, sole);
    }
    stage = createStage(Object.values(figures));
  } catch (error) {
    console.error(error);
    elements.count.textContent = "No motions loaded";
    showNotice(
      "The motion data did not load",
      "Build the data with <code>uv run scripts/build_site.py</code>. Then serve this directory with " +
      "<code>uv run scripts/serve.py</code>. A page that you open from the file system cannot read the data.",
    );
    return;
  }
  // Most motions share a few sets of driven joints, so find the groups of each set only once.
  const drivesOf = new Map();  // driven joints -> the groups that they drive in full
  for (const entry of index.motions) {
    if (!drivesOf.has(entry.active)) {
      const active = new Set(entry.active);
      drivesOf.set(entry.active, new Set(index.groups
        .filter(({ joints }) => joints.every((joint) => active.has(joint)))
        .map(({ title }) => title)));
    }
    entry.drives = drivesOf.get(entry.active);
    // Two sources can share a category name, so the source is part of the key.
    entry.categoryKey = `${entry.source}:${entry.category}`;
    entry.haystack = [entry.name, entry.clip ?? "", entry.category, index.sources[entry.source].title]
      .join(" ").toLowerCase();
    byName.set(entry.name, entry);
  }
  state.sources = new Set(Object.keys(index.sources));

  // The character is on show unless this visitor picked the robot.
  const picked = readStorage(FIGURE_KEY);
  showFigure(figures[picked] ? picked : figures.character ? "character" : "robot");
  if (figures.character) {
    for (const choice of document.querySelectorAll(".figure-choice")) choice.hidden = false;
    for (const button of document.querySelectorAll("[data-figure]")) {
      button.addEventListener("click", () => showFigure(button.dataset.figure));
    }
  }

  elements.search.addEventListener("input", () => {
    state.query = elements.search.value.toLowerCase().split(/\s+/).filter(Boolean);
    applyFilters();
  });
  elements.sort.addEventListener("change", () => {
    state.sort = elements.sort.value;
    applyFilters();
  });
  // A wider or narrower grid changes the size of each card, and thus each still frame.
  new ResizeObserver(redrawCards).observe(elements.grid);
  applyFilters();

  window.addEventListener("hashchange", openFromHash);
  openFromHash();
  requestAnimationFrame(loop);
}

main();
