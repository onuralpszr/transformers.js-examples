
import {
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    load_image,
    full,
} from '@huggingface/transformers';

async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter.features.has('shader-f16');
    } catch (e) {
        return false;
    }
}

/**
 * This class uses the Singleton pattern to ensure that only one instance of the model is loaded.
 */
class Paligemma2Singleton {
    static model_id = 'onnx-community/paligemma2-3b-pt-224';
    static async getInstance(progress_callback = null) {
        this.processor ??= AutoProcessor.from_pretrained(this.model_id);
        this.supports_fp16 ??= await hasFp16();
        this.model ??= PaliGemmaForConditionalGeneration.from_pretrained(this.model_id, {
            dtype: {
                embed_tokens: 'q8', // or 'q8'
                vision_encoder: 'q8', // or 'q4', 'q8'
                decoder_model_merged: 'q4', // or 'q4f16'
            },
            progress_callback,
        });

        return Promise.all([this.model, this.processor]);
    }
}


async function load() {
    self.postMessage({
        status: 'loading',
        data: 'Loading model...'
    });

    console.log('Loading model...');

    // Load the pipeline and save it for future use.
    const [model, processor] = await Paligemma2Singleton.getInstance(x => {
        // We also add a progress callback to the pipeline so that we can
        // track model loading.
        self.postMessage(x);
    });

    console.log('Loading model2...');

    self.postMessage({
        status: 'loading',
        data: 'Compiling shaders and warming up model...'
    });

    // Dummy text and vision inputs
    const text_inputs = "a";
    const pixel_values = full([1, 3, 768, 768], 0.0);
    const inputs = await processor(pixel_values, text_inputs);

    // Run model with dummy input to compile shaders
    await model.generate({
        ...inputs,
        max_new_tokens: 1,
    });

    self.postMessage({ status: 'ready' });
}


let raw_image;
let prompt;
async function run({ text, url, task }) {
    const [model, processor] = await Paligemma2Singleton.getInstance();

    // Read and preprocess image
    prompt =  task + ' ' + text;
    raw_image = await load_image(url);
    const start = performance.now();
    const inputs = await processor(raw_image, prompt);

    // Generate results
    const output = await model.generate({
        ...inputs,
        max_new_tokens: 100,
    });

    // Decode generated text
    const generated_ids = output.slice(null, [inputs.input_ids.dims[1], null]);

    // Post-process the generated text
    const result = processor.batch_decode(
        generated_ids,
        { skip_special_tokens: true },
    );

    const end = performance.now();

    self.postMessage({ status: 'complete', result, time: end - start });
}

// Listen for messages from the main thread
self.addEventListener('message', async (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'load':
            load();
            break;

        case 'run':
            run(data);
            break;

        case 'reset':
            raw_image,prompt = null;
            break;
    }
});
