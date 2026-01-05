import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import add_noise, apply_denoising, calculate_metrics, compare_algorithms

# Page config
st.set_page_config(
    page_title="Noise Reduction Website",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - HANYA BAGIAN INI YANG DIUBAH
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    /* Style untuk metric cards */
    div[data-testid="stMetric"] {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
    }
    div[data-testid="stMetric"] label {
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #4CAF50 !important;
        font-size: 32px !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: #FFD700 !important;
        font-size: 16px !important;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🖼️ Website Noise Reduction")
st.markdown("**Project Praktikum Pengolahan Citra Digital**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Mode selection
    mode = st.radio(
        "Mode",
        ["Single Algorithm", "Compare Algorithms", "Add Noise + Denoise"],
        help="Pilih mode operasi"
    )
    
    st.markdown("---")
    
    # Algorithm selection
    if mode != "Compare Algorithms":
        algorithm = st.selectbox(
            "Pilih Algoritma Denoising",
            [
                "Gaussian Blur",
                "Median Filter",
                "Bilateral Filter",
                "Non-Local Means",
                "Morphological Opening",
                "Morphological Closing"
            ]
        )
        
        # Parameters based on algorithm
        st.subheader("Parameters")
        
        if algorithm in ["Gaussian Blur", "Median Filter", "Morphological Opening", "Morphological Closing"]:
            kernel_size = st.slider("Kernel Size", 3, 15, 5, step=2)
            params = {'kernel_size': kernel_size}
        
        elif algorithm == "Bilateral Filter":
            d = st.slider("Diameter", 5, 15, 9, step=2)
            sigma_color = st.slider("Sigma Color", 10, 150, 75, step=5)
            sigma_space = st.slider("Sigma Space", 10, 150, 75, step=5)
            params = {'d': d, 'sigma_color': sigma_color, 'sigma_space': sigma_space}
        
        elif algorithm == "Non-Local Means":
            h = st.slider("Filter Strength (h)", 3, 20, 10)
            template_window = st.slider("Template Window Size", 5, 9, 7, step=2)
            search_window = st.slider("Search Window Size", 15, 25, 21, step=2)
            params = {'h': h, 'template_window': template_window, 'search_window': search_window}
    
    else:
        # Compare mode
        st.subheader("Algorithms to Compare")
        compare_algos = st.multiselect(
            "Pilih 2-4 algoritma",
            [
                "Gaussian Blur",
                "Median Filter",
                "Bilateral Filter",
                "Non-Local Means"
            ],
            default=["Gaussian Blur", "Median Filter", "Bilateral Filter"]
        )
        
        # Default params for comparison
        params = {
            'Gaussian Blur': {'kernel_size': 5},
            'Median Filter': {'kernel_size': 5},
            'Bilateral Filter': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
            'Non-Local Means': {'h': 10, 'template_window': 7, 'search_window': 21}
        }
    
    # Noise settings (for Add Noise mode)
    if mode == "Add Noise + Denoise":
        st.markdown("---")
        st.subheader("Noise Settings")
        noise_type = st.selectbox(
            "Jenis Noise",
            ["Gaussian", "Salt & Pepper", "Speckle", "Poisson"]
        )
        noise_intensity = st.slider("Noise Intensity", 1, 50, 20)

# Main content
uploaded_file = st.file_uploader(
    "📤 Upload Gambar",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload gambar dalam format JPG, PNG, atau BMP"
)

if uploaded_file is not None:
    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original
    original_image = image.copy()
    
    # Mode: Single Algorithm
    if mode == "Single Algorithm":
        st.subheader(f"Processing dengan {algorithm}")
        
        # Process
        with st.spinner('Processing...'):
            denoised = apply_denoising(image, algorithm, params)
        
        # Display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown(f"#### Denoised Image ({algorithm})")
            st.image(denoised, use_container_width=True)
        
        # Metrics
        st.markdown("---")
        st.subheader("📊 Quality Metrics")
        
        metrics = calculate_metrics(image, denoised)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("PSNR (dB)", f"{metrics['PSNR']:.2f}", help="Peak Signal-to-Noise Ratio - Higher is better")
        col2.metric("SSIM", f"{metrics['SSIM']:.4f}", help="Structural Similarity Index - Closer to 1 is better")
        col3.metric("MSE", f"{metrics['MSE']:.2f}", help="Mean Squared Error - Lower is better")
        
        # Download button
        st.markdown("---")
        result_pil = Image.fromarray(denoised)
        
        # Convert to bytes
        import io
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="⬇️ Download Hasil",
            data=byte_im,
            file_name=f"denoised_{algorithm.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
    
    # Mode: Compare Algorithms
    elif mode == "Compare Algorithms":
        if len(compare_algos) < 2:
            st.warning("⚠️ Pilih minimal 2 algoritma untuk dibandingkan!")
        else:
            st.subheader("Perbandingan Algoritma")
            
            with st.spinner('Processing multiple algorithms...'):
                results = compare_algorithms(image, compare_algos, params)
            
            # Display original
            st.markdown("#### Original Image")
            st.image(image, width=400)
            
            st.markdown("---")
            st.markdown("#### Hasil Denoising")
            
            # Display results in columns
            cols = st.columns(len(compare_algos))
            
            for idx, (algo_name, denoised_img) in enumerate(results.items()):
                with cols[idx]:
                    st.markdown(f"**{algo_name}**")
                    st.image(denoised_img, use_container_width=True)
                    
                    # Metrics for each
                    metrics = calculate_metrics(image, denoised_img)
                    
                    # Display metrics dengan spacing
                    st.markdown("")
                    st.metric("PSNR", f"{metrics['PSNR']:.2f} dB")
                    st.metric("SSIM", f"{metrics['SSIM']:.4f}")
                    st.metric("MSE", f"{metrics['MSE']:.2f}")
            
            # Metrics comparison table
            st.markdown("---")
            st.subheader("📊 Metrics Comparison Table")
            
            metrics_data = []
            for algo_name, denoised_img in results.items():
                metrics = calculate_metrics(image, denoised_img)
                metrics_data.append({
                    'Algorithm': algo_name,
                    'PSNR (dB)': f"{metrics['PSNR']:.2f}",
                    'SSIM': f"{metrics['SSIM']:.4f}",
                    'MSE': f"{metrics['MSE']:.2f}"
                })
            
            import pandas as pd
            df = pd.DataFrame(metrics_data)
            st.dataframe(df, use_container_width=True)
    
    # Mode: Add Noise + Denoise
    elif mode == "Add Noise + Denoise":
        st.subheader(f"Add {noise_type} Noise → Denoise dengan {algorithm}")
        
        with st.spinner('Adding noise and denoising...'):
            # Add noise
            noisy_image = add_noise(image, noise_type, noise_intensity)
            
            # Denoise
            denoised = apply_denoising(noisy_image, algorithm, params)
        
        # Display 3 columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Original")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown(f"#### With {noise_type} Noise")
            st.image(noisy_image, use_container_width=True)
        
        with col3:
            st.markdown(f"#### Denoised ({algorithm})")
            st.image(denoised, use_container_width=True)
        
        # Metrics
        st.markdown("---")
        st.subheader("📊 Quality Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original vs Noisy**")
            metrics_noisy = calculate_metrics(image, noisy_image)
            subcol1, subcol2, subcol3 = st.columns(3)
            subcol1.metric("PSNR", f"{metrics_noisy['PSNR']:.2f} dB")
            subcol2.metric("SSIM", f"{metrics_noisy['SSIM']:.4f}")
            subcol3.metric("MSE", f"{metrics_noisy['MSE']:.2f}")
        
        with col2:
            st.markdown("**Original vs Denoised**")
            metrics_denoised = calculate_metrics(image, denoised)
            subcol1, subcol2, subcol3 = st.columns(3)
            subcol1.metric("PSNR", f"{metrics_denoised['PSNR']:.2f} dB", 
                          delta=f"{metrics_denoised['PSNR'] - metrics_noisy['PSNR']:.2f}")
            subcol2.metric("SSIM", f"{metrics_denoised['SSIM']:.4f}",
                          delta=f"{metrics_denoised['SSIM'] - metrics_noisy['SSIM']:.4f}")
            subcol3.metric("MSE", f"{metrics_denoised['MSE']:.2f}",
                          delta=f"{metrics_denoised['MSE'] - metrics_noisy['MSE']:.2f}",
                          delta_color="inverse")

else:
    # Landing page
    st.info("👆 Upload gambar untuk memulai")
    
    st.markdown("### 📋 Fitur:")
    st.markdown("""
    - **Single Algorithm**: Test satu algoritma dengan parameter custom
    - **Compare Algorithms**: Bandingkan beberapa algoritma sekaligus
    - **Add Noise + Denoise**: Tambahkan noise artificial lalu lakukan denoising
    
    ### 🔧 Algoritma yang Tersedia:
    - Gaussian Blur
    - Median Filter
    - Bilateral Filter
    - Non-Local Means Denoising
    - Morphological Opening
    - Morphological Closing
    
    ### 📊 Metrics:
    - **PSNR** (Peak Signal-to-Noise Ratio): Mengukur quality, semakin tinggi semakin baik
    - **SSIM** (Structural Similarity Index): Mengukur similarity struktur, mendekati 1 semakin baik
    - **MSE** (Mean Squared Error): Mengukur error, semakin rendah semakin baik
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Project Praktikum Pengolahan Citra Digital | Streamlit + OpenCV"
    "</div>",
    unsafe_allow_html=True
)
