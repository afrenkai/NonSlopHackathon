import * as THREE from "three";
import { FpsMovement, PointerControls, SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

export function cameraWorldMatrix(matrix) {
  if (
    !Array.isArray(matrix) ||
    matrix.length !== 4 ||
    matrix.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== 4 ||
        row.some((value) => !Number.isFinite(value)),
    )
  ) {
    throw new Error("The reconstruction contains an invalid camera position.");
  }
  const cameraToWorld = new THREE.Matrix4().set(...matrix.flat());
  if (cameraToWorld.determinant() === 0) {
    throw new Error("The reconstruction contains an invalid camera position.");
  }
  return cameraToWorld.multiply(new THREE.Matrix4().makeRotationX(Math.PI));
}

export function applyCameraCalibration(camera, calibration, cameraId, width, height) {
  const intrinsics = calibration.intrinsics?.find((entry) => entry.camera_id === cameraId)?.matrix;
  if (
    !Array.isArray(intrinsics) ||
    intrinsics.length !== 3 ||
    intrinsics.some(
      (row) => !Array.isArray(row) || row.length !== 3 || row.some((value) => !Number.isFinite(value)),
    )
  ) {
    throw new Error("The reconstruction contains invalid camera calibration.");
  }
  const focalX = intrinsics[0][0];
  const focalY = intrinsics[1][1];
  const centerX = intrinsics[0][2];
  const centerY = intrinsics[1][2];
  const imageWidth = calibration.image_width ?? centerX * 2;
  const imageHeight = calibration.image_height ?? centerY * 2;
  if (
    focalX <= 0 || focalY <= 0 ||
    !Number.isFinite(imageWidth) || imageWidth <= 0 ||
    !Number.isFinite(imageHeight) || imageHeight <= 0
  ) {
    throw new Error("The reconstruction contains invalid camera calibration.");
  }
  const scale = Math.min(width / imageWidth, height / imageHeight);
  const principalX = centerX * scale + (width - imageWidth * scale) / 2;
  const principalY = centerY * scale + (height - imageHeight * scale) / 2;
  camera.aspect = width / height;
  camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(height / (2 * focalY * scale)));
  camera.projectionMatrix.makePerspective(
    -principalX * camera.near / (focalX * scale),
    (width - principalX) * camera.near / (focalX * scale),
    principalY * camera.near / (focalY * scale),
    -(height - principalY) * camera.near / (focalY * scale),
    camera.near,
    camera.far,
  );
  camera.projectionMatrix.elements[4] = -2 * intrinsics[0][1] * scale / width;
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}

export function cameraPath(cameraMatrices) {
  const positions = cameraMatrices.map((matrix) =>
    new THREE.Vector3().setFromMatrixPosition(matrix),
  );
  const bounds = new THREE.Box3().setFromPoints(positions);
  const radius = Math.max(bounds.getSize(new THREE.Vector3()).length() * 0.005, 0.015);
  const group = new THREE.Group();
  const markerGeometry = new THREE.SphereGeometry(radius, 10, 8);
  const markerMaterial = new THREE.MeshBasicMaterial({
    color: 0xf1a16e,
    transparent: true,
    opacity: 0.9,
    depthTest: false,
  });
  for (const position of positions) {
    const marker = new THREE.Mesh(markerGeometry, markerMaterial);
    marker.position.copy(position);
    group.add(marker);
  }
  if (positions.length > 1) {
    group.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(positions),
        new THREE.LineBasicMaterial({
          color: 0xf1a16e,
          transparent: true,
          opacity: 0.75,
          depthTest: false,
        }),
      ),
    );
  }
  return group;
}

async function initializeViewer(root) {
  const canvas = root.querySelector("#world-canvas");
  const status = root.querySelector("#world-status");
  const cameraCount = root.querySelector("#world-camera-count");
  const resetButton = root.querySelector("#world-reset");
  const showCameras = root.querySelector("#world-show-cameras");
  const worldUrl = `/jobs/${encodeURIComponent(root.dataset.jobId)}/world`;
  let renderer;
  let splats;
  let resizeObserver;

  function showError() {
    status.textContent =
      "This 3D visit could not be loaded. Return to your visits and try again.";
    status.classList.add("message", "message-error");
    status.setAttribute("role", "alert");
    root.setAttribute("aria-busy", "false");
    resetButton.disabled = true;
    showCameras.disabled = true;
    renderer?.setAnimationLoop(null);
  }

  try {
    const response = await fetch(`${worldUrl}/cameras`, {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error("The reconstruction cameras are unavailable.");
    }
    const cameraData = await response.json();
    if (!Array.isArray(cameraData.extrinsics) || cameraData.extrinsics.length === 0) {
      throw new Error("The reconstruction has no input camera positions.");
    }
    const cameraMatrices = cameraData.extrinsics.map((pose) =>
      cameraWorldMatrix(pose.matrix),
    );
    cameraCount.textContent = `${cameraMatrices.length} input camera ${cameraMatrices.length === 1 ? "position" : "positions"}`;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080d12);
    const camera = new THREE.PerspectiveCamera();
    camera.near = 0.01;
    camera.far = 1000;
    const initialMatrix = cameraMatrices[0];

    function resetView() {
      initialMatrix.decompose(camera.position, camera.quaternion, camera.scale);
      camera.up.set(0, 1, 0).applyQuaternion(camera.quaternion);
      camera.updateMatrixWorld(true);
      if (splats?.isInitialized) status.textContent = "Your 3D visit is ready. Select the scene to explore.";
    }

    resetView();
    renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    scene.add(new SparkRenderer({ renderer, preBlurAmount: 0.3, blurAmount: 0 }));
    const movement = new FpsMovement({ moveSpeed: 0.2 });
    const pointerControls = new PointerControls({ canvas });
    const path = cameraPath(cameraMatrices);
    path.visible = showCameras.checked;
    scene.add(path);
    splats = new SplatMesh({ url: `${worldUrl}/gaussians.ply`, lod: true });
    scene.add(splats);

    function resize() {
      const width = Math.max(canvas.clientWidth, 1);
      const height = Math.max(canvas.clientHeight, 1);
      renderer.setSize(width, height, false);
      applyCameraCalibration(camera, cameraData, cameraData.extrinsics[0].camera_id, width, height);
    }

    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    resize();
    canvas.addEventListener("pointerdown", () => {
      canvas.focus({ preventScroll: true });
    });
    canvas.addEventListener("keydown", (event) => {
      if (
        ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageUp", "PageDown"].includes(
          event.code,
        )
      ) {
        event.preventDefault();
      }
    });
    canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      showError();
    });
    showCameras.addEventListener("change", () => {
      path.visible = showCameras.checked;
    });
    resetButton.addEventListener("click", () => {
      resetView();
      canvas.focus({ preventScroll: true });
    });
    let previousTime;
    renderer.setAnimationLoop((time) => {
      const deltaTime = previousTime === undefined ? 0 : Math.min((time - previousTime) / 1000, 0.05);
      previousTime = time;
      movement.enable = document.activeElement === canvas;
      const moved = movement.update(deltaTime, camera);
      const pointed = pointerControls.update(deltaTime, camera);
      if (splats.isInitialized && deltaTime > 0 && (moved || pointed)) {
        const message = "Exploring the reconstructed space. Reset view to return to the photograph.";
        if (status.textContent !== message) status.textContent = message;
      }
      renderer.render(scene, camera);
    });

    await splats.initialized;
    status.textContent = "Your 3D visit is ready. Select the scene to explore.";
    root.setAttribute("aria-busy", "false");
    resetButton.disabled = false;
    showCameras.disabled = false;
  } catch (error) {
    console.error("Unable to load the 3D visit.", error);
    showError();
  }

  window.addEventListener("pagehide", (event) => {
    if (event.persisted) {
      return;
    }
    resizeObserver?.disconnect();
    renderer?.setAnimationLoop(null);
    splats?.dispose();
    renderer?.dispose();
  });
}

document.querySelectorAll("[data-world-viewer]").forEach((root) => {
  initializeViewer(root);
});
