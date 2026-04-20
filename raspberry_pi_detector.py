#!/usr/bin/env python3
"""
EdgeShield — Raspberry Pi V2 Dual Threat Detector (DNN + SNN)
Requires ARM-compatible PyTorch: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import sounddevice as sd
import librosa
import time
import json
import socket
import threading
import psutil
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import snntorch as snn
    from snntorch import surrogate
    SNN_AVAILABLE = True
except Exception:
    SNN_AVAILABLE = False
    print("WARNING: snntorch not available — SNN will be skipped")


class Config:
    BASE_DIR           = Path(__file__).parent
    MODELS_DIR         = BASE_DIR / 'models'
    LOG_DIR            = BASE_DIR / 'logs'
    DNN_MODEL_PATH     = MODELS_DIR / 'best_dnn_multimodal_v2.pth'
    SNN_MODEL_PATH     = MODELS_DIR / 'best_snn_multimodal_v2.pth'
    LAPTOP_IP          = '10.35.194.239'
    LAPTOP_PORT        = 5555
    SEND_VIDEO         = True
    DEVICE             = 'cpu'
    CAMERA_INDEX       = 0
    SAMPLE_RATE        = 16000
    AUDIO_DURATION     = 4.0
    CONFIDENCE_THRESHOLD = 0.5
    ALERT_CLASSES      = ['gun_shot', 'siren', 'car_horn']
    CLASSES = [
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
        'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
        'siren', 'street_music'
    ]
    TIMESTAMP          = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE           = LOG_DIR / f'detections_{TIMESTAMP}.log'
    PERFORMANCE_LOG    = LOG_DIR / 'performance_metrics.json'
    HISTORY_LOG        = LOG_DIR / f'detection_history_{TIMESTAMP}.json'


class AudioCompressorV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(128 * 4 * 4, 256)
        self.bn4   = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        return self.relu4(self.bn4(self.fc(x)))


class VisualCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.AvgPool2d(kernel_size=8)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(64 * 4 * 4, 512)
        self.bn3   = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
    def forward(self, x):
        x = self.pool1(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool2(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        return self.relu3(self.bn3(self.fc(x)))


class DNNClassifier(nn.Module):
    def __init__(self, input_size=768, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256); self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128);        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64);         self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.fc4(x)


class MultimodalDNN_V2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.audio_compressor  = AudioCompressorV2()
        self.visual_compressor = VisualCompressor()
        self.classifier        = DNNClassifier(input_size=768, num_classes=num_classes)
    def forward(self, mfcc, image):
        a = self.audio_compressor(mfcc)
        v = self.visual_compressor(image)
        return self.classifier(torch.cat([a, v], dim=1))


class SpikingClassifier(nn.Module):
    def __init__(self, input_size=768, num_classes=10, num_steps=25, beta=0.9, spike_grad=None):
        super().__init__()
        self.num_steps = num_steps
        self.fc1  = nn.Linear(input_size, 256); self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2  = nn.Linear(256, 128);        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3  = nn.Linear(128, 64);         self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc4  = nn.Linear(64, num_classes); self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
    def forward(self, x):
        m1=self.lif1.init_leaky(); m2=self.lif2.init_leaky()
        m3=self.lif3.init_leaky(); m4=self.lif4.init_leaky()
        spks = []
        for _ in range(self.num_steps):
            s1,m1=self.lif1(self.fc1(x),m1); s2,m2=self.lif2(self.fc2(s1),m2)
            s3,m3=self.lif3(self.fc3(s2),m3); s4,m4=self.lif4(self.fc4(s3),m4)
            spks.append(s4)
        return torch.stack(spks).sum(dim=0)


class MultimodalSNN_V2(nn.Module):
    def __init__(self, num_classes=10, num_steps=25, beta=0.9, spike_grad=None):
        super().__init__()
        self.audio_compressor  = AudioCompressorV2()
        self.visual_compressor = VisualCompressor()
        self.snn_classifier    = SpikingClassifier(768, num_classes, num_steps, beta, spike_grad)
    def forward(self, mfcc, image):
        a = self.audio_compressor(mfcc)
        v = self.visual_compressor(image)
        return self.snn_classifier(torch.cat([a, v], dim=1))


class AudioProcessor:
    def __init__(self, sample_rate=16000, duration=4.0):
        self.sr  = sample_rate
        self.tl  = int(sample_rate * duration)
        self.tf  = 130
    def process(self, audio):
        if len(audio) < self.tl:
            audio = np.pad(audio, (0, self.tl - len(audio)))
        else:
            audio = audio[:self.tl]
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=40, n_fft=2048, hop_length=512, fmax=8000)
        d1   = librosa.feature.delta(mfcc)
        d2   = librosa.feature.delta(mfcc, order=2)
        def norm(x):
            s = x.std(); return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)
        def fw(x):
            if x.shape[1] < self.tf: return np.pad(x, ((0,0),(0,self.tf-x.shape[1])))
            s = (x.shape[1]-self.tf)//2; return x[:,s:s+self.tf]
        out = np.stack([fw(norm(mfcc)), fw(norm(d1)), fw(norm(d2))], axis=0)
        return torch.FloatTensor(out).unsqueeze(0)


class VideoProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    def process(self, frame):
        return self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)


THREAT = {'gun_shot':'CRITICAL','siren':'HIGH','car_horn':'MEDIUM','dog_bark':'MEDIUM',
          'drilling':'LOW','jackhammer':'LOW','engine_idling':'LOW',
          'children_playing':'SAFE','air_conditioner':'SAFE','street_music':'SAFE'}

DISPLAY = {'air_conditioner':'Air Conditioner','car_horn':'Car Horn',
           'children_playing':'Children Playing','dog_bark':'Dog Bark',
           'drilling':'Drilling','engine_idling':'Engine Idling',
           'gun_shot':'GUN SHOT','jackhammer':'Jackhammer',
           'siren':'Siren','street_music':'Street Music'}


class DualThreatDetector:

    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        config.MODELS_DIR.mkdir(exist_ok=True)
        config.LOG_DIR.mkdir(exist_ok=True)

        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(config.LOG_FILE), logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)

        self.dnn_model = self._load_dnn()
        self.snn_model = self._load_snn()
        self.audio_processor = AudioProcessor(config.SAMPLE_RATE, config.AUDIO_DURATION)
        self.video_processor = VideoProcessor()

        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.camera.isOpened():
            raise RuntimeError(f"Cannot open camera {config.CAMERA_INDEX}")
        self.logger.info(f"Camera {config.CAMERA_INDEX} ready")

        self.metrics = {
            'dnn': {'total':0,'threats':0,'avg_latency':0,'per_class':{c:0 for c in config.CLASSES}},
            'snn': {'total':0,'threats':0,'avg_latency':0,'per_class':{c:0 for c in config.CLASSES}},
            'system': {'avg_cpu':0,'avg_memory':0,'total_cycles':0},
            'agreement': {'total':0,'agreed':0}
        }
        self.detection_history = []
        self.socket = None
        self._connect_to_laptop()

        self.logger.info("="*60)
        self.logger.info("EDGESHIELD V2 READY")
        self.logger.info(f"  DNN: {config.DNN_MODEL_PATH.name}")
        self.logger.info(f"  SNN: {config.SNN_MODEL_PATH.name if self.snn_model else 'SKIPPED'}")
        self.logger.info(f"  Audio: MFCC (3, 40, 130)")
        self.logger.info("="*60)

    def _load_dnn(self):
        p = self.config.DNN_MODEL_PATH
        if not p.exists(): raise FileNotFoundError(f"DNN not found: {p}")
        self.logger.info(f"Loading DNN: {p.name}")
        m = MultimodalDNN_V2(len(self.config.CLASSES))
        ck = torch.load(p, map_location=self.device, weights_only=False)
        m.load_state_dict(ck['model_state_dict']); m.to(self.device).eval()
        self.logger.info(f"  Acc:{ck.get('val_acc',0):.2f}% F1:{ck.get('val_f1',0):.2f}%")
        return m

    def _load_snn(self):
        if not SNN_AVAILABLE:
            self.logger.warning("snntorch unavailable — SNN skipped"); return None
        p = self.config.SNN_MODEL_PATH
        if not p.exists():
            self.logger.warning(f"SNN not found: {p.name} — skipped"); return None
        self.logger.info(f"Loading SNN: {p.name}")
        m = MultimodalSNN_V2(len(self.config.CLASSES), 25, 0.9, surrogate.fast_sigmoid(slope=25))
        ck = torch.load(p, map_location=self.device, weights_only=False)
        m.load_state_dict(ck['model_state_dict']); m.to(self.device).eval()
        self.logger.info(f"  Acc:{ck.get('val_acc',0):.2f}% F1:{ck.get('val_f1',0):.2f}%")
        return m

    def _connect_to_laptop(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(3)
            self.socket.connect((self.config.LAPTOP_IP, self.config.LAPTOP_PORT))
            self.logger.info(f"Connected to {self.config.LAPTOP_IP}")
        except Exception as e:
            self.logger.warning(f"Laptop connection failed: {e}")
            self.socket = None

    def capture_audio(self):
        try:
            a = sd.rec(int(self.config.AUDIO_DURATION*self.config.SAMPLE_RATE),
                       samplerate=self.config.SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait(); return a.flatten()
        except Exception as e:
            self.logger.error(f"Audio failed: {e}")
            return np.zeros(int(self.config.AUDIO_DURATION*self.config.SAMPLE_RATE))

    def _predict(self, model, mfcc, img):
        with torch.no_grad():
            probs = torch.softmax(model(mfcc, img).float(), dim=1)
            conf, pred = torch.max(probs, 1)
        return pred.item(), conf.item(), probs[0].cpu().numpy()

    def detect(self):
        t0 = time.time()
        audio_buf = [None]
        def rec(): audio_buf[0] = self.capture_audio()
        th = threading.Thread(target=rec); th.start()
        ret, frame = self.camera.read()
        th.join()
        if not ret: return None

        mfcc = self.audio_processor.process(audio_buf[0]).to(self.device)
        img  = self.video_processor.process(frame).to(self.device)

        t1 = time.time()
        dc, dconf, dprobs = self._predict(self.dnn_model, mfcc, img)
        dms = (time.time()-t1)*1000
        dcls = self.config.CLASSES[dc]

        scls, sconf, sms = None, None, None
        if self.snn_model:
            t1 = time.time()
            sc, sconf, _ = self._predict(self.snn_model, mfcc, img)
            sms = (time.time()-t1)*1000
            scls = self.config.CLASSES[sc]

        total_ms = (time.time()-t0)*1000
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        n   = self.metrics['system']['total_cycles'] + 1

        result = {
            'timestamp': datetime.now().isoformat(),
            'detection_number': n,
            'dnn': {'class':dcls,'threat':THREAT[dcls],'confidence':float(dconf),
                    'inference_ms':float(dms),'is_threat':dcls in self.config.ALERT_CLASSES,
                    'probabilities':{c:float(dprobs[i]) for i,c in enumerate(self.config.CLASSES)}},
            'snn': {'class':scls,'threat':THREAT[scls],'confidence':float(sconf),
                    'inference_ms':float(sms),'is_threat':scls in self.config.ALERT_CLASSES
                    } if scls else None,
            'agreed': (scls==dcls) if scls else None,
            'timing': {'dnn_ms':float(dms),'total_ms':float(total_ms)},
            'system': {'cpu':cpu,'memory':mem}
        }

        self._update_metrics(result, n)
        self.detection_history.append(result)
        self._log(result)
        if self.socket and self.config.SEND_VIDEO:
            self._send(result, frame)
        return result

    def _update_metrics(self, r, n):
        self.metrics['system']['total_cycles'] = n
        dm = self.metrics['dnn']
        dm['total'] += 1; dm['per_class'][r['dnn']['class']] += 1
        dm['avg_latency'] = (dm['avg_latency']*(n-1)+r['dnn']['inference_ms'])/n
        if r['dnn']['is_threat']: dm['threats'] += 1
        if r['snn']:
            sm = self.metrics['snn']
            sm['total'] += 1; sm['per_class'][r['snn']['class']] += 1
            sm['avg_latency'] = (sm['avg_latency']*(n-1)+r['snn']['inference_ms'])/n
            if r['snn']['is_threat']: sm['threats'] += 1
            self.metrics['agreement']['total'] += 1
            if r['agreed']: self.metrics['agreement']['agreed'] += 1
        self.metrics['system']['avg_cpu'] = (self.metrics['system']['avg_cpu']*(n-1)+r['system']['cpu'])/n
        self.metrics['system']['avg_memory'] = (self.metrics['system']['avg_memory']*(n-1)+r['system']['memory'])/n

    def _log(self, r):
        d = r['dnn']
        line = f"#{r['detection_number']:04d} | DNN: {d['threat']:8s} {DISPLAY[d['class']]:18s} ({d['confidence']:5.1%}, {d['inference_ms']:6.1f}ms)"
        if r['snn']:
            s = r['snn']
            line += f" | SNN: {s['threat']:8s} {DISPLAY[s['class']]:18s} ({s['confidence']:5.1%}, {s['inference_ms']:6.1f}ms) | {'AGREE' if r['agreed'] else 'MISMATCH'}"
        if d['is_threat']:
            self.logger.warning(line)
            self.logger.warning(f"  *** THREAT: {d['class'].upper()} ***")
        else:
            self.logger.info(line)

    def _send(self, result, frame):
        try:
            pkt = {'type':'detection','data':result}
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            pkt['frame'] = buf.tobytes().hex()
            self.socket.sendall((json.dumps(pkt)+'\n').encode('utf-8'))
        except Exception as e:
            self.logger.error(f"Send failed: {e}"); self._connect_to_laptop()

    def save_logs(self):
        with open(self.config.PERFORMANCE_LOG,'w') as f: json.dump(self.metrics,f,indent=4)
        with open(self.config.HISTORY_LOG,'w') as f: json.dump(self.detection_history,f,indent=2)

    def print_summary(self):
        n = self.metrics['system']['total_cycles']
        if n == 0: return
        self.logger.info("="*60)
        self.logger.info(f"SUMMARY — {n} cycles")
        self.logger.info(f"  DNN avg latency: {self.metrics['dnn']['avg_latency']:.1f}ms | threats: {self.metrics['dnn']['threats']}")
        if self.metrics['snn']['total'] > 0:
            self.logger.info(f"  SNN avg latency: {self.metrics['snn']['avg_latency']:.1f}ms | threats: {self.metrics['snn']['threats']}")
            ag = self.metrics['agreement']
            self.logger.info(f"  Agreement: {ag['agreed']}/{ag['total']} ({100*ag['agreed']/ag['total']:.1f}%)")
        self.logger.info(f"  CPU: {self.metrics['system']['avg_cpu']:.1f}% | RAM: {self.metrics['system']['avg_memory']:.1f}%")
        self.logger.info("="*60)

    def run(self, num_detections=None):
        self.logger.info("Detection started — Ctrl+C to stop")
        try:
            count = 0
            while True:
                self.detect(); count += 1
                if num_detections and count >= num_detections: break
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.logger.info("Stopping...")
        finally:
            self.save_logs(); self.print_summary()
            if self.camera: self.camera.release()
            if self.socket: self.socket.close()
            self.logger.info("Shutdown complete.")


def main():
    print("="*60)
    print("EDGESHIELD V2 — RASPBERRY PI DETECTOR")
    for name, path in [('DNN', Config.DNN_MODEL_PATH), ('SNN', Config.SNN_MODEL_PATH)]:
        status = f"({path.stat().st_size/1024/1024:.1f}MB) FOUND" if path.exists() else "NOT FOUND"
        print(f"  {name}: {path.name} — {status}")
    print()
    DualThreatDetector(Config).run()


if __name__ == "__main__":
    main()
