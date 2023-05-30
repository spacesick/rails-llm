from flask import Flask, request, jsonify
import infer

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt')
    seq_len = data.get('seq_len', 16)
    temperature = data.get('temperature', 0.8)
    filter_thres = data.get('filter_thres', 0.9)
    model = data.get('model', 'palm_1b_8k_v0')
    dtype = data.get('dtype', 'fp32')

    generated_response = infer.response(
        prompt,
        seq_len=seq_len,
        temperature=temperature,
        filter_thres=filter_thres,
        model=model,
        dtype=dtype
    )

    return jsonify(generated_response)

@app.route('/test', methods=['GET'])
def test():
    prompt = 'What is the meaning of life?'
    seq_len = request.args.get('seq_len', default=16, type=int)
    temperature = request.args.get('temperature', default=0.8, type=float)
    filter_thres = request.args.get('filter_thres', default=0.9, type=float)
    model = request.args.get('model', default='palm_1b_8k_v0', type=str)
    dtype = request.args.get('dtype', default='fp32', type=str)

    generated_response = infer.response(
        prompt,
        seq_len=seq_len,
        temperature=temperature,
        filter_thres=filter_thres,
        model=model,
        dtype=dtype
    )

    return jsonify(generated_response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
