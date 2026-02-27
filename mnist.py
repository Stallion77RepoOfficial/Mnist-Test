import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QSpinBox, QComboBox, QProgressBar)
from PyQt6.QtGui import QPainter, QPen, QColor, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Cihaz ayarı (Mac kullandığın için varsa mps yoksa cpu)
device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x, return_activations=False):
        a1 = self.relu(self.l1(x))
        a2 = self.relu(self.l2(a1))
        a3 = self.relu(self.l3(a2))
        out = self.l4(a3)
        if return_activations:
            return out, [a1, a2, a3, out]
        return out

def save_npz(model, path, img_size):
    data = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    data["img_size"] = np.array([img_size])
    np.savez(path, **data)

def load_npz(path):
    data = np.load(path)
    img_size = int(data["img_size"][0])
    model = Net(img_size * img_size)
    state = {k: torch.tensor(data[k]) for k in model.state_dict().keys()}
    model.load_state_dict(state)
    model.eval()
    return model, img_size

class TrainThread(QThread):
    progress = pyqtSignal(int)
    finished_train = pyqtSignal(object)

    def __init__(self, img_size, epochs):
        super().__init__()
        self.img_size = img_size
        self.epochs = epochs

    def run(self):
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(root=os.path.join(BASE_DIR, "data"), 
                                              train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        
        model = Net(self.img_size * self.img_size).to(device)
        opt = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        total_steps = len(loader) * self.epochs
        step = 0
        
        for e in range(self.epochs):
            for x, y in loader:
                x = x.view(-1, self.img_size * self.img_size).to(device)
                y = y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                
                step += 1
                if step % 20 == 0:
                    self.progress.emit(int((step / total_steps) * 100))
                    
        model.eval()
        self.progress.emit(100)
        self.finished_train.emit(model)

def center_image(img):
    img = img.convert("L")
    img_inv = ImageOps.invert(img) # Beyaz arka planı siyaha çevir (MNIST formatı)
    bbox = img_inv.getbbox()
    if not bbox: return img_inv
    
    digit = img_inv.crop(bbox)
    w, h = digit.size
    m = max(w, h) + 24 # Rakamın etrafına boşluk bırak
    new = Image.new("L", (m, m), 0)
    new.paste(digit, ((m - w) // 2, (m - h) // 2))
    return new

class NetworkVizWidget(QWidget):
    def __init__(self, w, h):
        super().__init__()
        self.setFixedSize(w, h)
        self.layer_sizes = [10, 8, 8, 8, 10]
        self.activations = [[0]*size for size in self.layer_sizes]

    def update_activations(self, acts):
        if not acts: return
        self.activations[0] = [np.random.rand() * 0.4 for _ in range(self.layer_sizes[0])]
        for i, act in enumerate(acts):
            act_np = act.detach().cpu().numpy()[0]
            if len(act_np) > 0:
                act_np = (act_np - act_np.min()) / (act_np.max() - act_np.min() + 1e-5)
                self.activations[i+1] = act_np[:self.layer_sizes[i+1]]
        self.update()

    def reset(self):
        self.activations = [[0]*size for size in self.layer_sizes]
        self.update()

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.fillRect(0, 0, self.width(), self.height(), QColor(25, 25, 25))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        margin_x, nodes = 40, []
        spacing_x = (self.width() - 2 * margin_x) / (len(self.layer_sizes) - 1)
        for i, size in enumerate(self.layer_sizes):
            layer_nodes = []
            x = margin_x + i * spacing_x
            spacing_y = self.height() / (size + 1)
            for j in range(size):
                layer_nodes.append((x, (j + 1) * spacing_y, self.activations[i][j]))
            nodes.append(layer_nodes)
        
        painter.setPen(QPen(QColor(70, 70, 70, 100), 1))
        for i in range(len(nodes) - 1):
            for n1 in nodes[i]:
                for n2 in nodes[i+1]:
                    painter.drawLine(int(n1[0]), int(n1[1]), int(n2[0]), int(n2[1]))
        
        for layer in nodes:
            for x, y, act in layer:
                intensity = int(act * 200) + 55
                painter.setBrush(QColor(intensity, intensity // 2, 0) if act > 0.05 else QColor(45, 45, 45))
                painter.setPen(QPen(QColor(200, 200, 200), 1))
                painter.drawEllipse(int(x - 7), int(y - 7), 14, 14)

class VizWidget(QWidget):
    def __init__(self, w, h):
        super().__init__()
        self.setFixedSize(w, h)
        self.probs = [0] * 10
        self.pred = 0
    def update_probs(self, pred, probs):
        self.pred, self.probs = pred, probs
        self.update()
    def reset(self):
        self.probs, self.pred = [0] * 10, 0
        self.update()
    def paintEvent(self, e):
        painter = QPainter(self)
        painter.fillRect(0, 0, self.width(), self.height(), QColor(35, 35, 35))
        bw = self.width() / 10
        for i, p in enumerate(self.probs):
            h = p * (self.height() - 60)
            x, y = i * bw + 6, self.height() - h - 35
            painter.setBrush(QColor(0, 255, 150) if i == self.pred else QColor(80, 80, 120))
            painter.drawRect(int(x), int(y), int(bw - 12), int(h))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(int(x + bw/4), int(self.height() - 12), str(i))

class DrawWidget(QWidget):
    def __init__(self, w, h, parent):
        super().__init__()
        self.parent = parent
        self.setFixedSize(w, h)
        self.image = QImage(w, h, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        self.last = None
        self.pil_img = Image.new("L", (w, h), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        
    def mousePressEvent(self, e): self.last = e.position().toPoint()
    def mouseMoveEvent(self, e):
        if self.last:
            p = e.position().toPoint()
            painter = QPainter(self.image)
            pen = QPen(Qt.GlobalColor.black, 24, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(self.last, p)
            painter.end()
            self.pil_draw.line([self.last.x(), self.last.y(), p.x(), p.y()], fill=0, width=24)
            self.last = p
            self.update()
            self.parent.realtime_predict()
    def mouseReleaseEvent(self, e): self.last = None
    def paintEvent(self, e):
        p = QPainter(self)
        p.drawImage(0, 0, self.image)
        p.setPen(QPen(QColor(120, 120, 120), 2))
        p.drawRect(0, 0, self.width()-1, self.height()-1)
    def clear(self):
        self.image.fill(Qt.GlobalColor.white)
        self.pil_img = Image.new("L", (self.width(), self.height()), "white")
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self.update()

class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST Visualizer")
        self.setStyleSheet("background-color: #1a1a1a; color: #eeeeee;")
        
        main_layout = QVBoxLayout()
        
        # Üst Panel (Grafikler)
        top_layout = QHBoxLayout()
        self.viz = VizWidget(450, 250)
        self.net_viz = NetworkVizWidget(450, 250)
        top_layout.addWidget(self.viz)
        top_layout.addWidget(self.net_viz)
        main_layout.addLayout(top_layout)
        
        # Orta Panel (Çizim)
        draw_layout = QHBoxLayout()
        self.draw = DrawWidget(320, 320, self)
        draw_layout.addStretch()
        draw_layout.addWidget(self.draw)
        draw_layout.addStretch()
        main_layout.addLayout(draw_layout)

        # ALT KISIM: Progress Bar (Klavuz altta bar şeklinde)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { background-color: #333; border-radius: 6px; }
            QProgressBar::chunk { background-color: #00ccff; border-radius: 6px; }
        """)
        main_layout.addWidget(QLabel("Training Progress:"))
        main_layout.addWidget(self.progress_bar)
        
        # Kontroller
        controls = QHBoxLayout()
        self.size = QSpinBox(); self.size.setRange(28, 64); self.size.setValue(28)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 20); self.epochs.setValue(8)
        self.model_list = QComboBox()
        self.refresh_models()
        
        self.btn_train = QPushButton("Train New Model")
        self.btn_load = QPushButton("Load Model")
        self.btn_clear = QPushButton("Clear Canvas")
        
        for w in [QLabel("Img Size:"), self.size, QLabel("Epochs:"), self.epochs, self.model_list, self.btn_train, self.btn_load, self.btn_clear]:
            controls.addWidget(w)
        main_layout.addLayout(controls)
        
        self.setLayout(main_layout)
        self.model = None
        self.img_size = 28
        
        self.btn_train.clicked.connect(self.start_training)
        self.btn_load.clicked.connect(self.load_selected)
        self.btn_clear.clicked.connect(self.reset_all)

    def refresh_models(self):
        self.model_list.clear()
        if os.path.exists(MODELS_DIR):
            self.model_list.addItems([f for f in os.listdir(MODELS_DIR) if f.endswith(".npz")])

    def start_training(self):
        self.btn_train.setEnabled(False)
        self.img_size = self.size.value()
        self.train_thread = TrainThread(self.img_size, self.epochs.value())
        self.train_thread.progress.connect(self.progress_bar.setValue)
        self.train_thread.finished_train.connect(self.on_training_finished)
        self.train_thread.start()

    def on_training_finished(self, trained_model):
        self.model = trained_model
        save_npz(self.model, os.path.join(MODELS_DIR, f"mnist_{self.img_size}px_{self.epochs.value()}ep.npz"), self.img_size)
        self.refresh_models()
        self.btn_train.setEnabled(True)

    def load_selected(self):
        name = self.model_list.currentText()
        if name:
            self.model, self.img_size = load_npz(os.path.join(MODELS_DIR, name))
            self.size.setValue(self.img_size)

    def realtime_predict(self):
        if not self.model: return
        # Görüntü işleme
        img = center_image(self.draw.pil_img)
        img = img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        # Önce Tensor'e çevir [1, H, W]
        t = transforms.ToTensor()(img)
        # Görüntü formundayken normalize et
        t = transforms.Normalize((0.1307,), (0.3081,))(t)
        # Sonra düzleştir [1, input_size]
        t = t.view(1, -1)
        
        with torch.no_grad():
            out, acts = self.model(t, return_activations=True)
            probs = torch.softmax(out, dim=1).numpy()[0]
            self.viz.update_probs(int(np.argmax(probs)), probs)
            self.net_viz.update_activations(acts)

    def reset_all(self):
        self.draw.clear(); self.viz.reset(); self.net_viz.reset()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec())
