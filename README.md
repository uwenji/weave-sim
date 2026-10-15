# WeaveSim Web

Simulation of bending-active woven timber (Kagome) gridshells in the browser: the laths are discrete elastic rods,
the crossings slide with Coulomb friction, solved by dynamic relaxation on the GPU (WebGPU) or on the CPU in double
precision.

**Open it:** https://uwenji.github.io/weave-sim/ (Chrome / Edge, Safari 26+; without WebGPU it uses the CPU solver)

- `web/` - the app; start with `web/HOW_TO_START.md`, details and validation in `web/README.md`
- `examples/` - the example scenes, meshes and the calibration example

Run it locally: `python web/serve.py`, then open http://localhost:8090/web/

Feedback is welcome - please open an issue.
