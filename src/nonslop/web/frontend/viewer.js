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
  const guidedCamera = root.querySelector("#world-guided");
  const playButton = root.querySelector("#world-play");
  const worldUrl = `/jobs/${encodeURIComponent(root.dataset.jobId)}/world`;
  let renderer;
  let splats;
  let pointerControls;
  let resizeObserver;
  const coverageMeshes = new Map();
  let playing = false;
  let playTime = 0;

  function showError() {
    status.textContent =
      "This 3D visit could not be loaded. Return to your visits and try again.";
    status.classList.add("message", "message-error");
    status.setAttribute("role", "alert");
    root.setAttribute("aria-busy", "false");
    resetButton.disabled = true;
    showCameras.disabled = true;
    guidedCamera.disabled = true;
    playButton.disabled = true;
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
    if (cameraData.photo_count > 0 && cameraData.generated_frame_count > 0) {
      cameraCount.textContent = `${cameraData.photo_count} ${cameraData.photo_count === 1 ? "photograph" : "photographs"} + ${cameraData.generated_frame_count} H3 views · ${cameraMatrices.length} cameras`;
    }

    const scene = new THREE.Scene();
    scene.background = cameraData.background_rgb
      ? new THREE.Color().setRGB(...cameraData.background_rgb, THREE.SRGBColorSpace)
      : new THREE.Color(0x080d12);
    const camera = new THREE.PerspectiveCamera();
    camera.near = 0.01;
    camera.far = Math.max(1000, (cameraData.scene_depth || 1) * 64);
    const initialMatrix = cameraMatrices[0];
    const inputPoses = cameraMatrices.map((matrix, index) => {
      const position = new THREE.Vector3();
      const quaternion = new THREE.Quaternion();
      matrix.decompose(position, quaternion, new THREE.Vector3());
      const cameraId = cameraData.extrinsics[index].camera_id;
      const intrinsics = cameraData.intrinsics.find((entry) => entry.camera_id === cameraId)?.matrix;
      return { position, quaternion, intrinsics };
    });
    const projectionCameraId = cameraData.extrinsics[0].camera_id;
    const projectionCalibration = {
      ...cameraData,
      intrinsics: [{ camera_id: projectionCameraId, matrix: inputPoses[0].intrinsics }],
    };
    const pathSpan = new THREE.Box3().setFromPoints(inputPoses.map((pose) => pose.position))
      .getSize(new THREE.Vector3()).length();
    const sceneDepth = cameraData.scene_depth || Math.max(pathSpan * 5, 1);
    const coverageRadius = Math.max(sceneDepth * 0.02, pathSpan * 0.2);
    const tourPoses = [inputPoses[0], ...inputPoses.slice(Math.max(cameraData.photo_count || 0, 1))];
    const tourDuration = Math.max(1, Number(root.dataset.videoDuration) || 8);

    function stopPlayback() {
      playing = false;
      playButton.textContent = "Play memory";
    }

    function clearPointerInertia() {
      pointerControls?.scroll.set(0, 0, 0);
      pointerControls?.rotateVelocity.set(0, 0, 0);
      pointerControls?.moveVelocity.set(0, 0, 0);
    }

    function updateProjection(intrinsics) {
      projectionCalibration.intrinsics[0].matrix = intrinsics;
      applyCameraCalibration(
        camera, projectionCalibration, projectionCameraId,
        Math.max(canvas.clientWidth, 1), Math.max(canvas.clientHeight, 1),
      );
    }

    function nearestPose() {
      let nearest = 0;
      let bestScore = Infinity;
      inputPoses.forEach((pose, index) => {
        const score = camera.position.distanceToSquared(pose.position)
          + sceneDepth * sceneDepth * 0.05 * (1 - Math.abs(camera.quaternion.dot(pose.quaternion)));
        if (score < bestScore) {
          bestScore = score;
          nearest = index;
        }
      });
      return nearest;
    }

    function updateCoverage(index) {
      if (!cameraData.coverage_available) return;
      if (!coverageMeshes.has(index)) {
        const cameraId = cameraData.extrinsics[index].camera_id;
        const mesh = new SplatMesh({ url: `${worldUrl}/coverage/${cameraId}.ply`, lod: false });
        mesh.visible = false;
        coverageMeshes.set(index, mesh);
        scene.add(mesh);
        mesh.initialized.catch((error) => {
          console.warn("Photographic gap fill could not be loaded.", error);
          scene.remove(mesh);
          mesh.dispose();
        });
      }
      const selected = coverageMeshes.get(index);
      if (selected?.isInitialized) {
        for (const mesh of coverageMeshes.values()) mesh.visible = mesh === selected;
      }
    }

    function resetView() {
      stopPlayback();
      clearPointerInertia();
      initialMatrix.decompose(camera.position, camera.quaternion, camera.scale);
      updateProjection(inputPoses[0].intrinsics);
      camera.up.set(0, 1, 0).applyQuaternion(camera.quaternion);
      camera.updateMatrixWorld(true);
      if (splats?.isInitialized) status.textContent = "Your 3D visit is ready. Select the scene to explore.";
    }

    resetView();
    renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    scene.add(new SparkRenderer({ renderer, preBlurAmount: 0.3, blurAmount: 0 }));
    const movement = new FpsMovement({ moveSpeed: Math.max(sceneDepth * 0.04, pathSpan * 0.1) });
    pointerControls = new PointerControls({ canvas });
    const path = cameraPath(cameraMatrices);
    path.visible = showCameras.checked;
    scene.add(path);
    splats = new SplatMesh({ url: `${worldUrl}/gaussians.ply`, lod: true });
    scene.add(splats);
    updateCoverage(0);

    const imageWidth = cameraData.image_width || cameraData.intrinsics[0].matrix[0][2] * 2;
    const imageHeight = cameraData.image_height || cameraData.intrinsics[0].matrix[1][2] * 2;
    canvas.style.width = "auto";
    canvas.style.height = "auto";
    canvas.style.maxWidth = "100%";
    canvas.style.maxHeight = "72vh";
    canvas.style.aspectRatio = `${imageWidth} / ${imageHeight}`;
    canvas.style.marginInline = "auto";
    canvas.width = imageWidth;
    canvas.height = imageHeight;

    function resize() {
      const width = Math.max(canvas.clientWidth, 1);
      const height = Math.max(canvas.clientHeight, 1);
      renderer.setSize(width, height, false);
      applyCameraCalibration(camera, projectionCalibration, projectionCameraId, width, height);
    }

    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    resize();
    canvas.addEventListener("pointerdown", () => {
      stopPlayback();
      canvas.focus({ preventScroll: true });
    });
    canvas.addEventListener("wheel", stopPlayback, { passive: true });
    canvas.addEventListener("keydown", (event) => {
      stopPlayback();
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
    playButton.addEventListener("click", () => {
      if (playing) stopPlayback();
      else {
        clearPointerInertia();
        playTime = 0;
        playing = true;
        playButton.textContent = "Pause memory";
      }
    });
    let previousTime;
    renderer.setAnimationLoop((time) => {
      const deltaTime = previousTime === undefined ? 0 : Math.min((time - previousTime) / 1000, 0.05);
      previousTime = time;
      movement.enable = !playing && document.activeElement === canvas;
      const moved = movement.update(deltaTime, camera);
      const pointed = !playing && pointerControls.update(deltaTime, camera);
      if (playing) {
        playTime = Math.min(playTime + deltaTime, tourDuration);
        const phase = playTime / tourDuration * (tourPoses.length - 1);
        const segment = Math.min(Math.floor(phase), tourPoses.length - 2);
        const fraction = THREE.MathUtils.smoothstep(phase - segment, 0, 1);
        camera.position.lerpVectors(tourPoses[segment].position, tourPoses[segment + 1].position, fraction);
        camera.quaternion.slerpQuaternions(tourPoses[segment].quaternion, tourPoses[segment + 1].quaternion, fraction);
        const startIntrinsics = tourPoses[segment].intrinsics;
        const endIntrinsics = tourPoses[segment + 1].intrinsics;
        updateProjection(startIntrinsics.map((row, rowIndex) => row.map(
          (value, columnIndex) => THREE.MathUtils.lerp(value, endIntrinsics[rowIndex][columnIndex], fraction),
        )));
        status.textContent = "Playing the reconstructed camera path.";
        if (playTime >= tourDuration) stopPlayback();
      }
      const closest = nearestPose();
      if (guidedCamera.checked && !playing) {
        const pose = inputPoses[closest];
        if (moved || pointed) updateProjection(pose.intrinsics);
        const offset = camera.position.clone().sub(pose.position);
        if (offset.length() > coverageRadius) camera.position.copy(pose.position).add(offset.setLength(coverageRadius));
        const angle = pose.quaternion.angleTo(camera.quaternion);
        const maxAngle = THREE.MathUtils.degToRad(15);
        if (angle > maxAngle) {
          const desired = camera.quaternion.clone();
          camera.quaternion.copy(pose.quaternion).slerp(desired, maxAngle / angle);
        }
      }
      updateCoverage(closest);
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
    guidedCamera.disabled = false;
    playButton.disabled = tourPoses.length < 2;
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
    for (const mesh of coverageMeshes.values()) mesh.dispose();
    renderer?.dispose();
  });
}

document.querySelectorAll("[data-world-viewer]").forEach((root) => {
  initializeViewer(root);
});
