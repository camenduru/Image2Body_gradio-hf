from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import concurrent.futures
import redis

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
from scripts.process_utils import *

from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

app = Flask(__name__)
# app.secret_key = 'super_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Redisクライアントの初期化（レート制限とキャッシュのため）
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# レート制限の設定
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# タスクキューの作成とサイズ制限
MAX_QUEUE_SIZE = 100
task_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
active_tasks = {}
task_futures = {}

# ThreadPoolExecutorの作成
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

class Task:
    def __init__(self, task_id, mode, weight1, weight2, file_data, client_ip):
        self.task_id = task_id
        self.mode = mode
        self.weight1 = weight1
        self.weight2 = weight2
        self.file_data = file_data
        self.cancel_flag = False
        self.client_ip = client_ip

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
            
        # タスク数をデクリメント
        client_ip = task.client_ip  # この行は Task クラスに client_ip 属性を追加する必要があります
        redis_client.decr(f'tasks:{client_ip}')
        
        update_queue_status('Task completed or cancelled')

def worker():
    while True:
        try:
            task = task_queue.get()
            if task.task_id in active_tasks:
                future = executor.submit(process_task, task)
                task_futures[task.task_id] = future
            update_queue_status(f'Task processing: {task.task_id}')
        except Exception as e:
            print(f"Worker error: {str(e)}")
        finally:
            # Ensure the task is always removed from the queue
            task_queue.task_done()

# ワーカースレッドの開始
threading.Thread(target=worker, daemon=True).start()

@app.route('/submit_task', methods=['POST'])
@limiter.limit("10 per minute")  # 1分間に10回までのリクエストに制限
def submit_task():
    if task_queue.full():
        return jsonify({'error': 'Task queue is full. Please try again later.'}), 503

    # クライアントIPアドレスを取得
    client_ip = get_remote_address()
    
    # 同一IPからの同時タスク数を制限
    if redis_client.get(f'tasks:{client_ip}') and int(redis_client.get(f'tasks:{client_ip}')) >= 2:
        return jsonify({'error': 'Maximum number of concurrent tasks reached'}), 429

    task_id = str(uuid.uuid4())
    file = request.files['file']
    mode = request.form.get('mode', 'refine')
    weight1 = float(request.form.get('weight1', 0.4))
    weight2 = float(request.form.get('weight2', 0.3))
    
    # ファイルタイプの制限
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type'}), 415

    # ファイルデータをバイト列として保存
    file_data = file.read()
    
    task = Task(task_id, mode, weight1, weight2, file_data, client_ip)
    task_queue.put(task)
    active_tasks[task_id] = task
    
    # 同一IPからのタスク数をインクリメント
    redis_client.incr(f'tasks:{client_ip}')
    redis_client.expire(f'tasks:{client_ip}', 3600)  # 1時間後に期限切れ
    
    update_queue_status(f'Task submitted: {task_id}')
    
    queue_size = task_queue.qsize()
    return jsonify({'task_id': task_id, 'queue_size': queue_size})

@app.route('/cancel_task/<task_id>', methods=['POST'])
def cancel_task(task_id):
    # クライアントIPアドレスを取得
    client_ip = get_remote_address()

    if task_id in active_tasks:
        task = active_tasks[task_id]
        # タスクの所有者を確認（IPアドレスで簡易的に判断）
        if task.client_ip != client_ip:
            return jsonify({'error': 'Unauthorized to cancel this task'}), 403
        task.cancel_flag = True
        if task_id in task_futures:
            task_futures[task_id].cancel()
            del task_futures[task_id]
        del active_tasks[task_id]
        # タスク数をデクリメント
        redis_client.decr(f'tasks:{client_ip}')
        update_queue_status('Task cancelled')
        return jsonify({'message': 'Task cancellation requested'})
    else:
        for task in list(task_queue.queue):
            if task.task_id == task_id and task.client_ip == client_ip:
                task.cancel_flag = True
                # タスク数をデクリメント
                redis_client.decr(f'tasks:{client_ip}')
                return jsonify({'message': 'Task cancellation requested for queued task'})
        return jsonify({'error': 'Task not found'}), 404

def get_active_task_order(task_id):
    return list(active_tasks.keys()).index(task_id) if task_id in active_tasks else None

# get_task_orderイベントハンドラー
@app.route('/get_task_order/<task_id>', methods=['GET'])
def handle_get_task_order(task_id):
    task_order = get_active_task_order(task_id)
    return jsonify({'task_order': task_order})

@socketio.on('connect')
def handle_connect(auth):
    # クライアント接続数の制限
    if redis_client.get('connected_clients') and int(redis_client.get('connected_clients')) > 100:
        return False  # 接続を拒否
    redis_client.incr('connected_clients')
    emit('queue_update', {'active_tasks': len(active_tasks), 'active_task_order': None})

@socketio.on('disconnect')
def handle_disconnect():
    redis_client.decr('connected_clients')

# Flaskルート
# ルートパスのGETリクエストに対するハンドラ
@app.route('/', methods=['GET'])
def root():
    return jsonify({"status": "ok", "message": "Server is running"}), 200

# process_refined のエンドポイント
@app.route('/process_refined', methods=['POST'])
def process_refined():
    file = request.files['file']
    weight1 = float(request.form.get('weight1', 0.4))
    weight2 = float(request.form.get('weight2', 0.3))
    
    image = ensure_rgb(Image.open(file.stream))
    sotai_image, sketch_image = process_image_as_base64(image, "refine", weight1, weight2)
    
    return jsonify({
        'sotai_image': sotai_image,
        'sketch_image': sketch_image
    })

@app.route('/process_original', methods=['POST'])
def process_original():
    file = request.files['file']
    
    image = ensure_rgb(Image.open(file.stream))
    sotai_image, sketch_image = process_image_as_base64(image, "original")
    
    return jsonify({
        'sotai_image': sotai_image,
        'sketch_image': sketch_image
    })

@app.route('/process_sketch', methods=['POST'])
def process_sketch():
    file = request.files['file']
    
    image = ensure_rgb(Image.open(file.stream))
    sotai_image, sketch_image = process_image_as_base64(image, "sketch")
    
    return jsonify({
        'sotai_image': sotai_image,
        'sketch_image': sketch_image
    })

# グローバルエラーハンドラー
@app.errorhandler(Exception)
def handle_exception(e):
    # ログにエラーを記録
    app.logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server options.')
    parser.add_argument('--use_local', action='store_true', help='Use local model')
    parser.add_argument('--use_gpu', action='store_true', help='Set to True to use GPU but if not available, it will use CPU')
    parser.add_argument('--use_dotenv', action='store_true', help='Use .env file for environment variables')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()
    
    # initialize(args.use_local, args.use_gpu, args.use_dotenv)
    port = int(os.environ['PORT']) if 'PORT' in os.environ else 5000
    print(f"Starting server on port {port}")
    server = pywsgi.WSGIServer(('0.0.0.0', port), app, handler_class=WebSocketHandler)
    server.serve_forever()
