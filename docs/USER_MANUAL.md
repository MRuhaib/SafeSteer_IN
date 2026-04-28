# SafeSteer-IN User Manual

## What This Tool Does

SafeSteer-IN compares a baseline language model response with a steered response that is intended to reduce harmful content in Indic-language prompts.

## Who Should Use It

- Non-technical evaluators who want to see the system behavior.
- Team members verifying the demo.
- Judges reviewing the backend pipeline and the steering effect.

## How To Run the Demo

1. Start the backend API.
2. Start the Gradio app.
3. Open the demo in your browser.
4. Enter a prompt or select one of the sample prompts.
5. Choose a language, category, and alpha if you want manual control.
6. Compare the baseline output and the steered output side by side.

## What You Should See

The interface should show:

- The input prompt.
- The detected or selected language.
- The detected or selected harm category.
- The baseline response.
- The steered response.
- Safety metadata such as risk score and steering status.

## API Summary

- `GET /health` checks whether the API is alive.
- `GET /ready` checks whether models/artifacts are loaded and serving is ready.
- `GET /slices` lists the slices that have steering vectors.
- `POST /generate` returns a baseline response only.
- `POST /steer` returns both baseline and steered outputs.

## Good Demo Prompts

Use a prompt that is clearly harmful, such as one involving hate speech, phishing, child exploitation, or misinformation. The system is designed to show a noticeable difference between the baseline and steered outputs in those cases.

## Troubleshooting

- If the UI does not load, confirm the backend is running.
- If steering is not applied, check whether vectors exist for the selected slice.
- If the system falls back to baseline output, the slice may be missing a vector file.
- If notifications fail, SMTP credentials may not be configured.

## Notes For Evaluators

The project has two phases:

1. Offline artifact generation: dataset, vectors, classifier, calibration, and evaluation.
2. Runtime inference: classifier routing plus steering during generation.

The documentation and codebase are structured so backend APIs can be used independently, and the UI and API are deployable as separate services via Docker Compose.
