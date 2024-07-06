from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import io
import os
import argparse
from PIL import Image
import torch
import gc
from peft import PeftModel

import queue
import threading
import uuid
import concurrent.futures
from process_utils import *

app = Flask(__name__)
# app.secret_key = 'super_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# タスクキューの作成
task_queue = queue.Queue()
active_tasks = {}
task_futures = {}

# ThreadPoolExecutorの作成
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

class Task:
    def __init__(self, task_id, mode, weight1, weight2, file_data):
        self.task_id = task_id
        self.mode = mode
        self.weight1 = weight1
        self.weight2 = weight2
        self.file_data = file_data
        self.cancel_flag = False

def update_queue_status(message=None):
    socketio.emit('queue_update', {'active_tasks': len(active_tasks), 'message': message})

def process_task(task):
    try:
        # ファイルデータをPIL Imageに変換
        image = Image.open(io.BytesIO(task.file_data))
        image = ensure_rgb(image)
        
        # キャンセルチェック
        if task.cancel_flag:
            return

        # 画像処理ロジックを呼び出す
        sotai_image, sketch_image = process_image_as_base64(image, task.mode, task.weight1, task.weight2)
        
        # キャンセルチェック
        if task.cancel_flag:
            return

        socketio.emit('task_complete', {
            'task_id': task.task_id, 
            'sotai_image': sotai_image, 
            'sketch_image': sketch_image
        })
    except Exception as e:
        if not task.cancel_flag:
            socketio.emit('task_error', {'task_id': task.task_id, 'error': str(e)})
    finally:
        if task.task_id in active_tasks:
            del active_tasks[task.task_id]
        if task.task_id in task_futures:
            del task_futures[task.task_id]
        update_queue_status('Task completed or cancelled')

def worker():
    while True:
        try:
            task = task_queue.get()
            if task.task_id in active_tasks:
                future = executor.submit(process_task, task)
                task_futures[task.task_id] = future
            update_queue_status(f'Task started: {task.task_id}')
        except Exception as e:
            print(f"Worker error: {str(e)}")
        finally:
            # Ensure the task is always removed from the queue
            task_queue.task_done()

# ワーカースレッドの開始
threading.Thread(target=worker, daemon=True).start()

@app.route('/submit_task', methods=['POST'])
def submit_task():
    task_id = str(uuid.uuid4())
    file = request.files['file']
    mode = request.form.get('mode', 'refine')
    weight1 = float(request.form.get('weight1', 0.4))
    weight2 = float(request.form.get('weight2', 0.3))
    
    # ファイルデータをバイト列として保存
    file_data = file.read()
    
    task = Task(task_id, mode, weight1, weight2, file_data)
    task_queue.put(task)
    active_tasks[task_id] = task
    
    update_queue_status(f'Task submitted: {task_id}')
    
    queue_size = task_queue.qsize()
    return jsonify({'task_id': task_id, 'queue_size': queue_size})

@app.route('/cancel_task/<task_id>', methods=['POST'])
def cancel_task(task_id):
    if task_id in active_tasks:
        task = active_tasks[task_id]
        task.cancel_flag = True
        if task_id in task_futures:
            task_futures[task_id].cancel()
            del task_futures[task_id]
        del active_tasks[task_id]
        update_queue_status('Task cancelled')
        return jsonify({'message': 'Task cancellation requested'})
    else:
        return jsonify({'message': 'Task not found or already completed'}), 404

def get_active_task_order(task_id):
    return list(active_tasks.keys()).index(task_id) if task_id in active_tasks else None

# get_task_orderイベントハンドラー
@app.route('/get_task_order/<task_id>', methods=['GET'])
def handle_get_task_order(task_id):
    task_order = get_active_task_order(task_id)
    return jsonify({'task_order': task_order})

@socketio.on('connect')
def handle_connect():
    emit('queue_update', {'active_tasks': len(active_tasks), 'active_task_order': None})

# Flaskルート
@app.route('/', methods=['GET', 'POST'])
def process_refined():
    if request.method == 'POST':
        file = request.files['file']
        weight1 = float(request.form.get('weight1', 0.4))
        weight2 = float(request.form.get('weight2', 0.3))
        
        image = ensure_rgb(Image.open(file.stream))
        sotai_image, sketch_image = process_image_as_base64(image, "refine", weight1, weight2)
        
        return jsonify({
            'sotai_image': sotai_image,
            'sketch_image': sketch_image
        })

@app.route('/process_original', methods=['GET', 'POST'])
def process_original():
    if request.method == 'POST':
        file = request.files['file']
        
        image = ensure_rgb(Image.open(file.stream))
        sotai_image, sketch_image = process_image_as_base64(image, "original")
        
        return jsonify({
            'sotai_image': sotai_image,
            'sketch_image': sketch_image
        })

@app.route('/process_sketch', methods=['GET', 'POST'])
def process_sketch():
    if request.method == 'POST':
        file = request.files['file']
        
        image = ensure_rgb(Image.open(file.stream))
        sotai_image, sketch_image = process_image_as_base64(image, "sketch")
        
        return jsonify({
            'sotai_image': sotai_image,
            'sketch_image': sketch_image
        })

# エラーハンドラー
@app.errorhandler(500)
def server_error(e):
    return jsonify(error=str(e)), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server options.')
    parser.add_argument('--local_model', type=bool, default=False, help='Use local model')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Set to True to use GPU but if not available, it will use CPU')
    args = parser.parse_args()
    
    initialize(local_model=args.local_model, use_gpu=args.use_gpu)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)