import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import pygame
import os
import threading
import time

class ImageVideoProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Processamento de Imagens e Vídeos")
        self.root.geometry("1200x800")
        
        # Inicializar pygame para áudio
        pygame.mixer.init()
        
        # Variáveis de estado
        self.current_image = None
        self.original_image = None
        self.video_capture = None
        self.is_video = False
        self.video_playing = False
        self.current_video_path = None
        self.tracker = None
        self.object_detected = False
        self.tracking_active = False
        self.detection_active = False
        self.sound_playing = False
        
        # Configurar estilo
        self.setup_styles()
        
        # Criar interface
        self.create_widgets()
        
    def setup_styles(self):
        """Configura os estilos da interface"""
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 9))
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('TLabelFrame', font=('Arial', 10, 'bold'))
        
    def create_widgets(self):
        """Cria todos os widgets da interface"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Notebook (abas)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Aba de Imagem
        self.image_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.image_frame, text="Processamento de Imagem")
        
        # Aba de Vídeo
        self.video_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.video_frame, text="Processamento de Vídeo")
        
        # Configurar as abas
        self.setup_image_tab()
        self.setup_video_tab()
        
    def setup_image_tab(self):
        """Configura a aba de processamento de imagem"""
        # Frame esquerdo para controles
        control_frame = ttk.Frame(self.image_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Frame direito para exibição
        display_frame = ttk.Frame(self.image_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ===== CONTROLES DE ARQUIVO =====
        file_frame = ttk.LabelFrame(control_frame, text="Arquivo", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="Carregar Imagem", 
                  command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Salvar Imagem", 
                  command=self.save_image).pack(fill=tk.X, pady=2)
        
        # ===== CONVERSÕES =====
        convert_frame = ttk.LabelFrame(control_frame, text="Conversões", padding=10)
        convert_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(convert_frame, text="Tons de Cinza", 
                  command=self.convert_grayscale).pack(fill=tk.X, pady=2)
        ttk.Button(convert_frame, text="Negativo", 
                  command=self.convert_negative).pack(fill=tk.X, pady=2)
        ttk.Button(convert_frame, text="Binária (Otsu)", 
                  command=self.convert_binary).pack(fill=tk.X, pady=2)
        
        # ===== FILTROS =====
        filter_frame = ttk.LabelFrame(control_frame, text="Filtros", padding=10)
        filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(filter_frame, text="Filtro da Média", 
                  command=self.apply_mean_filter).pack(fill=tk.X, pady=2)
        ttk.Button(filter_frame, text="Filtro da Mediana", 
                  command=self.apply_median_filter).pack(fill=tk.X, pady=2)
        ttk.Button(filter_frame, text="Detector de Bordas Canny", 
                  command=self.apply_canny).pack(fill=tk.X, pady=2)
        
        # ===== OPERAÇÕES MORFOLÓGICAS =====
        morph_frame = ttk.LabelFrame(control_frame, text="Operações Morfológicas", padding=10)
        morph_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(morph_frame, text="Erosão", 
                  command=self.apply_erosion).pack(fill=tk.X, pady=2)
        ttk.Button(morph_frame, text="Dilatação", 
                  command=self.apply_dilation).pack(fill=tk.X, pady=2)
        ttk.Button(morph_frame, text="Abertura", 
                  command=self.apply_opening).pack(fill=tk.X, pady=2)
        ttk.Button(morph_frame, text="Fechamento", 
                  command=self.apply_closing).pack(fill=tk.X, pady=2)
        
        # ===== ANÁLISE =====
        analysis_frame = ttk.LabelFrame(control_frame, text="Análise", padding=10)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_frame, text="Mostrar Histograma", 
                  command=self.show_histogram).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Calcular Métricas", 
                  command=self.calculate_metrics).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Contar Objetos", 
                  command=self.count_objects).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Restaurar Original", 
                  command=self.restore_original).pack(fill=tk.X, pady=2)
        
        # ===== ÁREA DE EXIBIÇÃO DA IMAGEM =====
        self.image_label = ttk.Label(display_frame, text="Imagem será exibida aqui\n\nClique em 'Carregar Imagem' para começar", 
                                   background='white', anchor=tk.CENTER, justify=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # ===== BARRA DE STATUS =====
        self.status_var = tk.StringVar()
        self.status_var.set("Pronto para carregar imagem")
        status_bar = ttk.Label(display_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, style='TLabel')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_video_tab(self):
        """Configura a aba de processamento de vídeo"""
        # Frame esquerdo para controles
        control_frame = ttk.Frame(self.video_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Frame direito para exibição
        display_frame = ttk.Frame(self.video_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ===== CONTROLES DE FONTE DE VÍDEO =====
        source_frame = ttk.LabelFrame(control_frame, text="Fonte de Vídeo", padding=10)
        source_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(source_frame, text="Carregar Vídeo", 
                  command=self.load_video).pack(fill=tk.X, pady=2)
        ttk.Button(source_frame, text="Acessar Câmera", 
                  command=self.access_camera).pack(fill=tk.X, pady=2)
        ttk.Button(source_frame, text="Parar Vídeo", 
                  command=self.stop_video).pack(fill=tk.X, pady=2)
        
        # ===== CONVERSÕES PARA VÍDEO =====
        convert_frame = ttk.LabelFrame(control_frame, text="Conversões em Vídeo", padding=10)
        convert_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(convert_frame, text="Tons de Cinza", 
                  command=lambda: self.apply_video_operation("grayscale")).pack(fill=tk.X, pady=2)
        ttk.Button(convert_frame, text="Negativo", 
                  command=lambda: self.apply_video_operation("negative")).pack(fill=tk.X, pady=2)
        ttk.Button(convert_frame, text="Binária (Otsu)", 
                  command=lambda: self.apply_video_operation("binary")).pack(fill=tk.X, pady=2)
        
        # ===== FILTROS PARA VÍDEO =====
        filter_frame = ttk.LabelFrame(control_frame, text="Filtros em Vídeo", padding=10)
        filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(filter_frame, text="Filtro da Média", 
                  command=lambda: self.apply_video_operation("mean")).pack(fill=tk.X, pady=2)
        ttk.Button(filter_frame, text="Filtro da Mediana", 
                  command=lambda: self.apply_video_operation("median")).pack(fill=tk.X, pady=2)
        ttk.Button(filter_frame, text="Detector de Bordas Canny", 
                  command=lambda: self.apply_video_operation("canny")).pack(fill=tk.X, pady=2)
        
        # ===== OPERAÇÕES MORFOLÓGICAS PARA VÍDEO =====
        morph_frame = ttk.LabelFrame(control_frame, text="Operações Morfológicas em Vídeo", padding=10)
        morph_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(morph_frame, text="Erosão", 
                  command=lambda: self.apply_video_operation("erosion")).pack(fill=tk.X, pady=2)
        ttk.Button(morph_frame, text="Dilatação", 
                  command=lambda: self.apply_video_operation("dilation")).pack(fill=tk.X, pady=2)
        ttk.Button(morph_frame, text="Abertura", 
                  command=lambda: self.apply_video_operation("opening")).pack(fill=tk.X, pady=2)
        ttk.Button(morph_frame, text="Fechamento", 
                  command=lambda: self.apply_video_operation("closing")).pack(fill=tk.X, pady=2)
        
        # ===== OPERAÇÕES ESPECÍFICAS DE VÍDEO =====
        video_ops_frame = ttk.LabelFrame(control_frame, text="Operações de Vídeo", padding=10)
        video_ops_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(video_ops_frame, text="Rastreamento de Objeto", 
                  command=self.object_tracking).pack(fill=tk.X, pady=2)
        ttk.Button(video_ops_frame, text="Detecção de Microfone", 
                  command=self.detect_microphone).pack(fill=tk.X, pady=2)
        ttk.Button(video_ops_frame, text="Reproduzir Vídeo Normal", 
                  command=self.play_video_normal).pack(fill=tk.X, pady=2)
        
        # ===== ÁREA DE EXIBIÇÃO DO VÍDEO =====
        self.video_label = ttk.Label(display_frame, text="Vídeo será exibido aqui\n\nUse 'Carregar Vídeo' ou 'Acessar Câmera' para começar", 
                                   background='black', anchor=tk.CENTER, foreground='white', justify=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # ===== BARRA DE STATUS DO VÍDEO =====
        self.video_status_var = tk.StringVar()
        self.video_status_var.set("Pronto para carregar vídeo ou acessar câmera")
        video_status_bar = ttk.Label(display_frame, textvariable=self.video_status_var, 
                                    relief=tk.SUNKEN, style='TLabel')
        video_status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # =========================================================================
    # MÉTODOS IMPLEMENTADOS: CARREGAR IMAGEM, VÍDEO, CÂMERA E REPRODUZIR VÍDEO
    # =========================================================================
    
    def load_image(self):
        """Método para carregar imagem - IMPLEMENTADO"""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Ler a imagem usando OpenCV
                self.original_image = cv2.imread(file_path)
                if self.original_image is not None:
                    self.current_image = self.original_image.copy()
                    self.is_video = False
                    
                    # Exibir informações da imagem
                    height, width = self.current_image.shape[:2]
                    channels = self.current_image.shape[2] if len(self.current_image.shape) == 3 else 1
                    
                    self.display_image(self.current_image)
                    self.status_var.set(f"Imagem carregada: {os.path.basename(file_path)} - {width}x{height} - {channels} canal(is)")
                else:
                    messagebox.showerror("Erro", "Não foi possível carregar a imagem. Verifique se o arquivo é válido.")
                    
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar imagem: {str(e)}")
        
    def load_video(self):
        """Método para carregar vídeo - IMPLEMENTADO"""
        file_path = filedialog.askopenfilename(
            title="Selecionar Vídeo",
            filetypes=[
                ("Vídeos", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Parar qualquer vídeo anterior
                self.stop_video()
                
                # Carregar o novo vídeo
                self.video_capture = cv2.VideoCapture(file_path)
                if self.video_capture.isOpened():
                    self.current_video_path = file_path
                    self.is_video = True
                    
                    # Obter informações do vídeo
                    fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                    frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    # Mostrar primeiro frame
                    ret, frame = self.video_capture.read()
                    if ret:
                        self.current_image = frame
                        self.display_video_frame(frame)
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset para início
                    
                    info_text = f"Vídeo: {os.path.basename(file_path)} - {width}x{height} - {fps:.1f} FPS - {duration:.1f}s"
                    self.video_status_var.set(info_text)
                    
                    messagebox.showinfo("Vídeo Carregado", 
                                      f"Vídeo carregado com sucesso!\n\n"
                                      f"Arquivo: {os.path.basename(file_path)}\n"
                                      f"Resolução: {width}x{height}\n"
                                      f"FPS: {fps:.1f}\n"
                                      f"Duração: {duration:.1f} segundos\n"
                                      f"Frames: {frame_count}")
                else:
                    messagebox.showerror("Erro", "Não foi possível carregar o vídeo. Verifique se o arquivo é válido.")
                    
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar vídeo: {str(e)}")
    
    def access_camera(self):
        """Método para acessar câmera - IMPLEMENTADO"""
        try:
            # Parar qualquer vídeo anterior
            self.stop_video()
            
            # Tentar acessar a câmera
            self.video_capture = cv2.VideoCapture(0)
            
            if not self.video_capture.isOpened():
                # Tentar câmera alternativa
                self.video_capture = cv2.VideoCapture(1)
            
            if self.video_capture.isOpened():
                self.is_video = True
                self.current_video_path = "camera"
                
                # Testar se a câmera funciona
                ret, frame = self.video_capture.read()
                if ret:
                    self.video_status_var.set("Câmera acessada com sucesso! Use 'Reproduzir Vídeo Normal' para visualizar")
                    messagebox.showinfo("Câmera", "Câmera acessada com sucesso! Use 'Reproduzir Vídeo Normal' para visualizar a câmera ao vivo.")
                else:
                    messagebox.showerror("Erro", "Câmera acessada mas não está fornecendo imagens.")
                    self.stop_video()
                    
            else:
                messagebox.showerror("Erro", "Não foi possível acessar nenhuma câmera. Verifique se a câmera está conectada e disponível.")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao acessar câmera: {str(e)}")
    
    def play_video_normal(self):
        """Método para reproduzir vídeo normal - IMPLEMENTADO"""
        if self.video_capture is None or not self.video_capture.isOpened():
            messagebox.showwarning("Aviso", "Nenhum vídeo carregado ou câmera não acessada")
            return
        
        # Se for um arquivo de vídeo, voltar ao início
        if self.current_video_path != "camera":
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.video_playing = True
        self.video_status_var.set("Reproduzindo vídeo - Pressione 'Q' na janela para parar")
        
        def video_loop():
            """Loop principal de reprodução de vídeo"""
            while self.video_playing and self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                
                if ret:
                    # Exibir o frame
                    self.display_video_frame(frame)
                    
                    # Adicionar informações no frame se for uma janela separada
                    if self.current_video_path == "camera":
                        cv2.putText(frame, "Câmera - Pressione Q para sair", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        cv2.putText(frame, f"Frame: {current_frame}/{total_frames} - Pressione Q para sair", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Mostrar em uma janela separada do OpenCV
                    cv2.imshow("Reprodução de Vídeo", frame)
                    
                    # Controlar a velocidade de reprodução
                    if self.current_video_path == "camera":
                        delay = 1  # Câmera em tempo real
                    else:
                        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                        delay = max(1, int(1000 / fps)) if fps > 0 else 30
                    
                    # Verificar se o usuário pressionou 'q' para sair
                    if cv2.waitKey(delay) & 0xFF == ord('q'):
                        break
                else:
                    # Fim do vídeo (apenas para arquivos)
                    if self.current_video_path != "camera":
                        break
            
            # Limpar após o loop
            cv2.destroyAllWindows()
            self.video_playing = False
            self.video_status_var.set("Reprodução finalizada")
            
            # Se era um arquivo de vídeo, resetar para o início
            if self.current_video_path != "camera" and self.video_capture:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_capture.read()
                if ret:
                    self.display_video_frame(frame)
        
        # Executar o loop de vídeo em uma thread separada
        video_thread = threading.Thread(target=video_loop)
        video_thread.daemon = True
        video_thread.start()

    # =========================================================================
    # CONVERSÕES BÁSICAS IMPLEMENTADAS (IMAGEM, VÍDEO E CÂMERA)
    # =========================================================================
    
    def convert_grayscale(self):
        """Método para converter para tons de cinza - IMPLEMENTADO"""
        if self.current_image is not None and not self.is_video:
            # Converter para tons de cinza
            if len(self.current_image.shape) == 3:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            self.display_image(self.current_image)
            self.status_var.set("Imagem convertida para tons de cinza")
        elif self.is_video:
            messagebox.showinfo("Info", "Use as conversões de vídeo na aba de Vídeo para processar vídeos")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para processar")
    
    def convert_negative(self):
        """Método para converter para negativo - IMPLEMENTADO"""
        if self.current_image is not None and not self.is_video:
            # Converter para negativo
            self.current_image = 255 - self.current_image
            self.display_image(self.current_image)
            self.status_var.set("Imagem convertida para negativo")
        elif self.is_video:
            messagebox.showinfo("Info", "Use as conversões de vídeo na aba de Vídeo para processar vídeos")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para processar")
    
    def convert_binary(self):
        """Método para converter para binária (Otsu) - IMPLEMENTADO"""
        if self.current_image is not None and not self.is_video:
            # Converter para binária usando método de Otsu
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image
            
            # Aplicar limiarização de Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.current_image = binary
            self.display_image(self.current_image)
            self.status_var.set("Imagem convertida para binária (Otsu)")
        elif self.is_video:
            messagebox.showinfo("Info", "Use as conversões de vídeo na aba de Vídeo para processar vídeos")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para processar")
    
    def apply_video_operation(self, operation):
        """Método para aplicar operação em vídeo - IMPLEMENTADO"""
        if self.video_capture is None or not self.video_capture.isOpened():
            messagebox.showwarning("Aviso", "Nenhum vídeo carregado ou câmera não acessada")
            return
        
        # Se for um arquivo de vídeo, voltar ao início
        if self.current_video_path != "camera":
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.video_playing = True
        operation_names = {
            "grayscale": "Tons de Cinza",
            "negative": "Negativo", 
            "binary": "Binária (Otsu)",
            "mean": "Filtro da Média",
            "median": "Filtro da Mediana",
            "canny": "Detector de Bordas Canny",
            "erosion": "Erosão",
            "dilation": "Dilatação",
            "opening": "Abertura",
            "closing": "Fechamento"
        }
        
        self.video_status_var.set(f"Aplicando {operation_names[operation]} - Pressione 'Q' para sair")
        
        def video_operation_loop():
            """Loop principal para operações de vídeo"""
            while self.video_playing and self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                
                if ret:
                    # Aplicar a operação especificada
                    processed_frame = self.apply_operation_to_frame(frame, operation)
                    
                    # Exibir o frame processado
                    self.display_video_frame(processed_frame)
                    
                    # Adicionar informações no frame
                    if self.current_video_path == "camera":
                        cv2.putText(processed_frame, f"{operation_names[operation]} - Câmera - Pressione Q para sair", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        cv2.putText(processed_frame, f"{operation_names[operation]} - Frame: {current_frame}/{total_frames} - Pressione Q para sair", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Mostrar em uma janela separada do OpenCV
                    cv2.imshow(f"Vídeo - {operation_names[operation]}", processed_frame)
                    
                    # Controlar a velocidade de reprodução
                    if self.current_video_path == "camera":
                        delay = 1  # Câmera em tempo real
                    else:
                        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                        delay = max(1, int(1000 / fps)) if fps > 0 else 30
                    
                    # Verificar se o usuário pressionou 'q' para sair
                    if cv2.waitKey(delay) & 0xFF == ord('q'):
                        break
                else:
                    # Fim do vídeo (apenas para arquivos)
                    if self.current_video_path != "camera":
                        break
            
            # Limpar após o loop
            cv2.destroyAllWindows()
            self.video_playing = False
            self.video_status_var.set(f"{operation_names[operation]} finalizado")
            
            # Se era um arquivo de vídeo, resetar para o início
            if self.current_video_path != "camera" and self.video_capture:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_capture.read()
                if ret:
                    self.display_video_frame(frame)
        
        # Executar o loop de operação de vídeo em uma thread separada
        video_thread = threading.Thread(target=video_operation_loop)
        video_thread.daemon = True
        video_thread.start()
    
    def apply_operation_to_frame(self, frame, operation):
        """Aplica uma operação específica a um frame"""
        try:
            if operation == "grayscale":
                # Converter para tons de cinza
                if len(frame.shape) == 3:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return frame
                
            elif operation == "negative":
                # Converter para negativo
                return 255 - frame
                
            elif operation == "binary":
                # Converter para binária usando Otsu
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary
                
            elif operation == "mean":
                # Aplicar filtro da média
                return cv2.blur(frame, (5, 5))
                
            elif operation == "median":
                # Aplicar filtro da mediana
                return cv2.medianBlur(frame, 5)
                
            elif operation == "canny":
                # Aplicar detector de bordas Canny
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                return cv2.Canny(gray, 50, 150)
                
            elif operation == "erosion":
                # Aplicar erosão
                kernel = np.ones((5, 5), np.uint8)
                return cv2.erode(frame, kernel, iterations=1)
                
            elif operation == "dilation":
                # Aplicar dilatação
                kernel = np.ones((5, 5), np.uint8)
                return cv2.dilate(frame, kernel, iterations=1)
                
            elif operation == "opening":
                # Aplicar abertura
                kernel = np.ones((5, 5), np.uint8)
                return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
                
            elif operation == "closing":
                # Aplicar fechamento
                kernel = np.ones((5, 5), np.uint8)
                return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
                
            else:
                return frame
                
        except Exception as e:
            print(f"Erro ao aplicar operação {operation}: {str(e)}")
            return frame

    def restore_original(self):
        """Método para restaurar imagem original - IMPLEMENTADO"""
        if self.original_image is not None and not self.is_video:
            self.current_image = self.original_image.copy()
            self.display_image(self.current_image)
            self.status_var.set("Imagem original restaurada")
        elif self.is_video:
            messagebox.showinfo("Info", "Para vídeos, use 'Parar Vídeo' e recarregue o vídeo")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para restaurar")
    
    def stop_video(self):
        """Método para parar vídeo - IMPLEMENTADO"""
        self.video_playing = False
        self.tracking_active = False
        self.detection_active = False
        
        # Pequena pausa para garantir que as threads parem
        time.sleep(0.1)
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        cv2.destroyAllWindows()
        self.video_status_var.set("Vídeo parado")
        
        # Limpar a exibição do vídeo
        self.video_label.configure(
            text="Vídeo será exibido aqui\n\nUse 'Carregar Vídeo' ou 'Acessar Câmera' para começar",
            background='black', 
            foreground='white'
        )
        self.video_label.image = None

    # =========================================================================
    # MÉTODOS AUXILIARES PARA EXIBIÇÃO
    # =========================================================================
    
    def display_image(self, image):
        """Exibe uma imagem na interface"""
        if image is None:
            return
            
        try:
            # Redimensionar imagem para caber no display
            h, w = image.shape[:2]
            max_width = 800
            max_height = 600
            
            if w > max_width or h > max_height:
                scale = min(max_width/w, max_height/h)
                new_w, new_h = int(w*scale), int(h*scale)
                image = cv2.resize(image, (new_w, new_h))
            
            # Converter para formato adequado para tkinter
            if len(image.shape) == 3:
                # Converter BGR para RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Imagem em tons de cinza
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Converter para ImageTk
            pil_image = Image.fromarray(image_rgb)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Atualizar a label
            self.image_label.configure(image=tk_image, text="")
            self.image_label.image = tk_image
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exibir imagem: {str(e)}")
    
    def display_video_frame(self, frame):
        """Exibe um frame de vídeo na interface"""
        if frame is None:
            return
            
        try:
            # Redimensionar frame para caber no display
            h, w = frame.shape[:2]
            max_width = 800
            max_height = 600
            
            if w > max_width or h > max_height:
                scale = min(max_width/w, max_height/h)
                new_w, new_h = int(w*scale), int(h*scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Converter para RGB se necessário
            if len(frame.shape) == 2:  # Imagem em tons de cinza
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:  # Imagem colorida
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Converter para ImageTk
            pil_image = Image.fromarray(frame_rgb)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Atualizar a label
            self.video_label.configure(image=tk_image, text="")
            self.video_label.image = tk_image
            
        except Exception as e:
            print(f"Erro ao exibir frame: {str(e)}")

    # =========================================================================
    # MÉTODOS NÃO IMPLEMENTADOS (MANTIDOS COMO PLACEHOLDER)
    # =========================================================================
    
    def save_image(self):
        if self.current_image is not None and not self.is_video:
            file_path = filedialog.asksaveasfilename(
                title="Salvar Imagem",
                defaultextension=".png",
                filetypes=[
                    ("PNG", "*.png"),
                    ("JPEG", "*.jpg"),
                    ("BMP", "*.bmp"),
                    ("Todos os arquivos", "*.*")
                ]
            )
            if file_path:
                cv2.imwrite(file_path, self.current_image)
                self.status_var.set(f"Imagem salva em: {file_path}")
                messagebox.showinfo("Sucesso", "Imagem salva com sucesso!")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem para salvar")
        
    def apply_mean_filter(self):
        if self.current_image is not None and not self.is_video:
            kernel_size = 5  # Pode tornar configurável
            self.current_image = cv2.blur(self.current_image, (kernel_size, kernel_size))
            self.display_image(self.current_image)
            self.status_var.set(f"Filtro da média aplicado (kernel {kernel_size}x{kernel_size})")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para processar")

        
    def apply_median_filter(self):
        if self.current_image is not None and not self.is_video:
            kernel_size = 5  # Pode tornar configurável
            self.current_image = cv2.medianBlur(self.current_image, kernel_size)
            self.display_image(self.current_image)
            self.status_var.set(f"Filtro da mediana aplicado (kernel {kernel_size})")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para processar")
        
    def apply_canny(self):
        if self.current_image is not None and not self.is_video:
            # Converter para tons de cinza se necessário
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image
            
            # Aplicar Canny
            edges = cv2.Canny(gray, 100, 200)  # Thresholds podem ser configuráveis
            self.current_image = edges
            self.display_image(self.current_image)
            self.status_var.set("Detector de bordas Canny aplicado")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para processar")
        
    def apply_erosion(self):
        if self.current_image is not None and not self.is_video:
            kernel = np.ones((5,5), np.uint8)
            self.current_image = cv2.erode(self.current_image, kernel, iterations=1)
            self.display_image(self.current_image)
            self.status_var.set("Erosão aplicada")

    def apply_dilation(self):
        if self.current_image is not None and not self.is_video:
            kernel = np.ones((5,5), np.uint8)
            self.current_image = cv2.dilate(self.current_image, kernel, iterations=1)
            self.display_image(self.current_image)
            self.status_var.set("Dilatação aplicada")

    def apply_opening(self):
        if self.current_image is not None and not self.is_video:
            kernel = np.ones((5,5), np.uint8)
            self.current_image = cv2.morphologyEx(self.current_image, cv2.MORPH_OPEN, kernel)
            self.display_image(self.current_image)
            self.status_var.set("Abertura aplicada")

    def apply_closing(self):
        if self.current_image is not None and not self.is_video:
            kernel = np.ones((5,5), np.uint8)
            self.current_image = cv2.morphologyEx(self.current_image, cv2.MORPH_CLOSE, kernel)
            self.display_image(self.current_image)
            self.status_var.set("Fechamento aplicada")

    def show_histogram(self):
        if self.current_image is not None and not self.is_video:
            if len(self.current_image.shape) == 3:
                # Imagem colorida - mostrar histogramas para cada canal
                colors = ('b', 'g', 'r')
                plt.figure(figsize=(10, 4))
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([self.current_image], [i], None, [256], [0, 256])
                    plt.plot(hist, color=color)
                    plt.xlim([0, 256])
                plt.title('Histograma - Colorido')
                plt.xlabel('Valores de Pixel')
                plt.ylabel('Frequência')
            else:
                # Imagem em tons de cinza
                plt.figure(figsize=(10, 4))
                hist = cv2.calcHist([self.current_image], [0], None, [256], [0, 256])
                plt.plot(hist, color='black')
                plt.title('Histograma - Tons de Cinza')
                plt.xlabel('Valores de Pixel')
                plt.ylabel('Frequência')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            self.status_var.set("Histograma exibido")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para processar")
        
    def calculate_metrics(self):
        if self.current_image is not None and not self.is_video:
            # Verificar se a imagem é binária, se não, converter
            if len(self.current_image.shape) == 3 or np.max(self.current_image) > 1:
                # Converter para binária usando Otsu
                if len(self.current_image.shape) == 3:
                    gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.current_image
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                binary = self.current_image
            
            # Encontrar contornos
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Calcular métricas para o maior contorno (ou todos)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Área
                area = cv2.contourArea(largest_contour)
                
                # Perímetro
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Diâmetro (diâmetro do círculo com mesma área)
                diameter = 2 * np.sqrt(area / np.pi)
                
                # Exibir resultados
                result_text = f"Métricas do Objeto:\n\nÁrea: {area:.2f} pixels\nPerímetro: {perimeter:.2f} pixels\nDiâmetro: {diameter:.2f} pixels"
                messagebox.showinfo("Métricas da Imagem Binária", result_text)
                self.status_var.set("Métricas calculadas")
            else:
                messagebox.showwarning("Aviso", "Nenhum objeto encontrado na imagem binária")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para processar")
        
    def count_objects(self):
        if self.current_image is not None and not self.is_video:
            # Garantir que a imagem é binária
            if len(self.current_image.shape) == 3 or np.max(self.current_image) > 1:
                if len(self.current_image.shape) == 3:
                    gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.current_image
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                binary = self.current_image
            
            # Algoritmo de crescimento de região (flood fill)
            def region_growing_count(img):
                height, width = img.shape
                visited = np.zeros_like(img, dtype=bool)
                object_count = 0
                
                # Direções dos vizinhos (8-conectividade)
                directions = [(-1, -1), (-1, 0), (-1, 1),
                            (0, -1),           (0, 1),
                            (1, -1),  (1, 0),  (1, 1)]
                
                def grow_region(x, y, label):
                    stack = [(x, y)]
                    visited[y, x] = True
                    
                    while stack:
                        cx, cy = stack.pop()
                        for dx, dy in directions:
                            nx, ny = cx + dx, cy + dy
                            if (0 <= nx < width and 0 <= ny < height and 
                                not visited[ny, nx] and img[ny, nx] == 255):
                                visited[ny, nx] = True
                                stack.append((nx, ny))
                
                # Percorrer a imagem
                for y in range(height):
                    for x in range(width):
                        if img[y, x] == 255 and not visited[y, x]:
                            object_count += 1
                            grow_region(x, y, object_count)
                
                return object_count
            
            # Contar objetos
            count = region_growing_count(binary)
            messagebox.showinfo("Contagem de Objetos", f"Número de objetos encontrados: {count}")
            self.status_var.set(f"Objetos contados: {count}")
            
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para processar")
        
    def object_tracking(self):
        print("Iniciando detecção avançada de rostos e olhos...")
        
        self.stop_video()
        time.sleep(0.5)

        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                messagebox.showerror("Erro", "Não foi possível acessar a câmera")
                return
                
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao acessar câmera: {e}")
            return

        # Carregar classificadores
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            if face_cascade.empty() or eye_cascade.empty():
                messagebox.showerror("Erro", "Classificadores não encontrados")
                return
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar classificadores: {e}")
            return

        self.video_playing = True
        self.detection_active = True
        self.current_video_path = "camera"

        def advanced_detection_loop():
            frame_count = 0
            
            while self.video_playing and self.detection_active and self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                frame_count += 1
                
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detectar rostos
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                processed_frame = frame.copy()
                face_count = len(faces)
                
                for (x, y, w, h) in faces:
                    # Desenhar rosto
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Região de interesse para olhos (parte superior do rosto)
                    roi_gray = gray[y:y + int(h/2), x:x + w]
                    roi_color = processed_frame[y:y + int(h/2), x:x + w]
                    
                    # Detectar olhos dentro do rosto
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                    
                    for (ex, ey, ew, eh) in eyes:
                        # Desenhar olhos
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                        cv2.putText(roi_color, "Olho", (ex, ey-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # Informações na tela
                cv2.putText(processed_frame, f"Rostos: {face_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(processed_frame, "Pressione 'Q' para sair", 
                        (10, processed_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                self.display_video_frame(processed_frame)
                cv2.imshow("Detecção de Rostos e Olhos", processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cv2.destroyAllWindows()
            self.video_playing = False
            self.detection_active = False
            if self.video_capture:
                self.video_capture.release()
            
            print(f"Detecção avançada finalizada. Frames: {frame_count}")
            self.video_status_var.set("Detecção avançada finalizada")
        
        advanced_thread = threading.Thread(target=advanced_detection_loop)
        advanced_thread.daemon = True
        advanced_thread.start()
            
    def detect_microphone(self):
        print("Iniciando detecção de microfone...")
    
        if self.current_video_path is None:
            messagebox.showwarning("Aviso", "Nenhum vídeo carregado")
            return

        # Parar qualquer vídeo anterior
        self.stop_video()
        time.sleep(0.5)  # Dar tempo para liberar recursos

        # Recriar o video_capture com configurações diferentes
        try:
            if self.current_video_path == "camera":
                self.video_capture = cv2.VideoCapture(0)
                # Configurações para câmera
                self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                # Para arquivos de vídeo, tentar diferentes backends
                self.video_capture = cv2.VideoCapture(self.current_video_path)
                
            if not self.video_capture.isOpened():
                messagebox.showerror("Erro", "Não foi possível abrir o vídeo/câmera")
                return
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao abrir vídeo: {e}")
            return

        # Carregar templates
        print("Carregando templates...")
        microphone_templates = self.load_microphone_templates()
        print(f"Templates carregados: {len(microphone_templates)}")
        
        if not microphone_templates:
            messagebox.showwarning("Aviso", "Não foram encontrados templates do microfone")
            return

        # Inicializar detector - usar ORB que é mais rápido e robusto
        print("Inicializando detector ORB...")
        detector = cv2.ORB_create(nfeatures=500)
        
        # Extrair características dos templates
        template_descriptors = []
        
        for i, template in enumerate(microphone_templates):
            # Redimensionar templates para tamanho consistente
            template = cv2.resize(template, (100, 100))
            kp, des = detector.detectAndCompute(template, None)
            if des is not None:
                template_descriptors.append(des)
                print(f"Template {i+1}: {len(kp)} keypoints")
            else:
                print(f"Template {i+1}: Não foi possível extrair características")
        
        if not template_descriptors:
            messagebox.showerror("Erro", "Não foi possível extrair características dos templates")
            return

        # Configurar matcher
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.video_playing = True
        self.detection_active = True
        self.sound_playing = False
        self.microphone_detected = False

        def detection_loop():
            frame_count = 0
            detection_count = 0
            
            while self.video_playing and self.detection_active:
                try:
                    ret, frame = self.video_capture.read()
                    frame_count += 1
                    
                    if not ret:
                        print("Fim do vídeo ou erro na leitura do frame")
                        break
                        
                    # Pular frames para melhor performance (processar 1 a cada 3 frames)
                    if frame_count % 3 != 0:
                        continue

                    # Reduzir resolução para melhor performance
                    frame = cv2.resize(frame, (640, 480))
                    processed_frame = frame.copy()
                    
                    # 1. PRÉ-FILTRAGEM POR COR VERMELHA
                    red_mask = self.detect_red_color(frame)
                    red_regions = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    red_regions = red_regions[0] if len(red_regions) == 2 else red_regions[1]
                    
                    # 2. DETECÇÃO DE CARACTERÍSTICAS
                    microphone_found = False
                    best_match_score = 0
                    best_match_location = None
                    
                    for contour in red_regions:
                        if cv2.contourArea(contour) < 300:  # Aumentei a área mínima
                            continue
                        
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Verificar proporção para filtrar formas não retangulares
                        aspect_ratio = w / h
                        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                            continue
                        
                        # Expandir ROI
                        padding = 15
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(frame.shape[1] - x, w + 2 * padding)
                        h = min(frame.shape[0] - y, h + 2 * padding)
                        
                        roi = frame[y:y+h, x:x+w]
                        
                        if roi.size == 0:
                            continue
                        
                        # Redimensionar ROI para tamanho consistente
                        roi = cv2.resize(roi, (100, 100))
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        
                        # Detectar características
                        kp_frame, des_frame = detector.detectAndCompute(roi_gray, None)
                        
                        if des_frame is not None and len(des_frame) > 10:
                            for template_des in template_descriptors:
                                try:
                                    matches = matcher.match(template_des, des_frame)
                                    matches = sorted(matches, key=lambda x: x.distance)
                                    
                                    # Pegar melhores matches (30% dos melhores)
                                    good_matches = matches[:int(len(matches) * 0.3)]
                                    match_score = len(good_matches)
                                    
                                    if match_score > best_match_score:
                                        best_match_score = match_score
                                        best_match_location = (x, y, w, h)
                                        microphone_found = True
                                        
                                except Exception as e:
                                    continue
                    
                    # 3. VERIFICAR DETECÇÃO
                    detection_threshold = 15  # Aumentei o threshold
                    
                    if microphone_found and best_match_score > detection_threshold:
                        detection_count += 1
                        if not self.sound_playing:
                            try:
                                self.play_detection_sound()
                                self.sound_playing = True
                                self.microphone_detected = True
                                print(f"🎤 Microfone detectado! Score: {best_match_score}")
                            except Exception as e:
                                print(f"Erro ao tocar som: {e}")
                        
                        # Desenhar detecção
                        x, y, w, h = best_match_location
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(processed_frame, f"MICROFONE (Score: {best_match_score})", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        if self.sound_playing:
                            self.stop_detection_sound()
                            self.sound_playing = False
                            self.microphone_detected = False
                    
                    # Exibir informações
                    status_text = f"Frame: {frame_count} | Microfone: {'DETECTADO' if self.microphone_detected else 'Nao detectado'} | Score: {best_match_score}"
                    cv2.putText(processed_frame, status_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    

                    
                    # Exibir na interface
                    self.display_video_frame(processed_frame)
                    
                    # Mostrar em janela separada
                    cv2.imshow("Detecção de Microfone Vermelho", processed_frame)
                    
                    # Controle de velocidade
                    delay = 1 if self.current_video_path == "camera" else 30
                    key = cv2.waitKey(delay) & 0xFF
                    if key == ord('q'):
                        print("Detecção interrompida pelo usuário")
                        break
                        
                except Exception as e:
                    print(f"Erro no processamento do frame: {e}")
                    continue
            
            # Limpeza final
            cv2.destroyAllWindows()
            self.video_playing = False
            self.detection_active = False
            if self.sound_playing:
                self.stop_detection_sound()
            
            print(f"✅ Detecção finalizada. Frames: {frame_count}, Detecções: {detection_count}")
            self.video_status_var.set("Detecção de microfone finalizada")
        
        # Executar em thread
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()

    def detect_red_color(self, frame):
        """Detectar regiões vermelhas na imagem"""
        # Converter para HSV para melhor detecção de cor
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Definir ranges para vermelho (HSV)
        # Vermelho tem dois ranges porque fica nos extremos do espectro
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        # Criar máscaras
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Operações morfológicas para limpar a máscara
        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        return red_mask

    def load_microphone_templates(self):
        """Carregar templates do microfone vermelho"""
        templates = []
        
        # Aqui você deve carregar suas imagens do microfone
        # Por enquanto, vou criar templates sintéticos - substitua por suas imagens reais
        try:
            # Exemplo: carregar de um diretório
            template_files = [
                "microphone_template1.jpg",
                "microphone_template2.jpg", 
                "microphone_template3.jpg",
                "microphone_template4.jpg",
                "microphone_template5.jpg",
                "microphone_template6.jpg",
                "microphone_template7.jpg",
                "microphone_template8.jpg",
                "microphone_template9.jpg",
                "microphone_template10.jpg",
                "microphone_template11.jpg",
                "microphone_template12.jpg",
                "microphone_template13.jpg",
                "microphone_template14.jpg",
                "microphone_template15.jpg",
                "microphone_template16.jpg"
            ]
            
            for file in template_files:
                if os.path.exists(file):
                    template = cv2.imread(file)
                    if template is not None:
                        # Converter para tons de cinza
                        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                        templates.append(template_gray)
            
            # Se não encontrou arquivos, criar um template simples (remova isso quando tiver suas imagens)
            if not templates:
                # Template sintético - SUBSTITUA POR IMAGENS REAIS DO SEU MICROFONE
                synthetic_template = np.zeros((50, 30), dtype=np.uint8)
                cv2.rectangle(synthetic_template, (10, 5), (20, 45), 255, -1)  # Cabo
                cv2.circle(synthetic_template, (15, 15), 10, 255, -1)  # Cabeça
                templates.append(synthetic_template)
                
        except Exception as e:
            print(f"Erro ao carregar templates: {e}")
        
        return templates

    def play_detection_sound(self):
        """Tocar música quando microfone for detectado"""
        try:
            pygame.mixer.music.load("musga.mp3")
            pygame.mixer.music.play(-1)  # -1 para tocar em loop
            print("Música iniciada: musga.mp3")
        except Exception as e:
            print(f"Erro ao tocar música: {e}")

    def stop_detection_sound(self):
        """Parar música quando microfone não for mais detectado"""
        try:
            pygame.mixer.music.stop()
            print("Música parada")
        except Exception as e:
            print(f"Erro ao parar música: {e}")

    def __del__(self):
        """Destruidor - limpeza de recursos"""
        self.stop_video()
        cv2.destroyAllWindows()

# =========================================================================
# EXECUÇÃO PRINCIPAL
# =========================================================================

if __name__ == "__main__":
    root = tk.Tk()
    
    def on_closing():
        """Função chamada ao fechar a janela"""
        root.quit()
        root.destroy()
    
    # Criar e executar a aplicação
    app = ImageVideoProcessor(root)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Centralizar a janela
    root.eval('tk::PlaceWindow . center')
    
    root.mainloop()
