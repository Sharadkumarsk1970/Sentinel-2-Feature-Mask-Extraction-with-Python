import os
import rasterio
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
from rasterio.warp import reproject, Resampling
from matplotlib.patches import Patch

# ============================================================
# --- Index Calculation Functions ---
# ============================================================

def calculate_index(numerator, denominator):
    np.seterr(divide='ignore', invalid='ignore')
    return np.nan_to_num((numerator - denominator) / (numerator + denominator), nan=0.0)

def extract_features(index_array, threshold, mode='above'):
    return (index_array > threshold).astype(np.uint8) if mode == 'above' else (index_array < threshold).astype(np.uint8)

def normalize(array):
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    return (array - array_min) / (array_max - array_min + 1e-6)

# ============================================================
# --- RGB Stretch Function (NEW) ---
# ============================================================

def stretch_rgb(rgb, lower=2, upper=98):
    """Apply percentile stretch to RGB composite for better visualization."""
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):  # R, G, B channels
        p_low, p_high = np.percentile(rgb[:, :, i], (lower, upper))
        out[:, :, i] = np.clip((rgb[:, :, i] - p_low) / (p_high - p_low + 1e-6), 0, 1)
    return out

# ============================================================
# --- Band Reading and Resampling ---
# ============================================================

def read_and_resample_band(path, ref_shape=None, ref_transform=None, ref_crs=None):
    print(f"📥 Reading band: {os.path.basename(path)}")
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)

        if ref_shape is not None:
            dst = np.empty(ref_shape, dtype=np.float32)
            reproject(
                source=band,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear
            )
            return dst
        else:
            return band, src.transform, src.crs, band.shape

# ============================================================
# --- Search All Resolutions for Band ---
# ============================================================

def find_band_path_any_resolution(safe_folder, band_code):
    pattern = os.path.join(
        safe_folder, "GRANULE", "*", "IMG_DATA", "*", f"*{band_code}*.jp2"
    )
    print(f"🔍 Searching for band {band_code} with pattern: {pattern}")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"❌ {band_code} band not found in any resolution.")
    print(f"✅ Found {band_code} at: {matches[0]}")
    return matches[0]

# ============================================================
# --- 🌈 RGB Composite Function ---
# ============================================================

def plot_rgb_composite(bands, name, save_plot, output_dir):
    """
    Plot an RGB composite with percentile stretch.
    """
    combos = {
        "Natural Color": ("B04", "B03", "B02"),  # Red, Green, Blue
        "False Color": ("B08", "B04", "B03"),   # NIR, Red, Green
    }

    if name not in combos:
        print(f"⚠️ {name} not in combos")
        return

    b1, b2, b3 = combos[name]
    rgb = np.dstack((bands[b1], bands[b2], bands[b3])).astype(np.float32)

    # Apply percentile stretch (instead of global min-max)
    rgb = stretch_rgb(rgb)

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title(f"{name} Composite")
    plt.axis("off")

    if save_plot:
        output_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_composite.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"💾 {name} composite saved to: {output_path}")
    else:
        plt.show()
    plt.close()

# ============================================================
# --- Main Processing Function ---
# ============================================================

def process_and_plot(safe_folder, save_plot=True):
    t0 = time.time()

    print("📂 Locating band files...")
    band_codes = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12']
    band_paths = {code: find_band_path_any_resolution(safe_folder, code) for code in band_codes}
    print(f"✅ Band files located in {time.time() - t0:.2f} seconds")

    print("📥 Reading and resampling all bands to match B02 (Blue)...")
    ref_band, ref_transform, ref_crs, ref_shape = read_and_resample_band(band_paths['B02'])
    bands = {'B02': ref_band}

    for code in band_codes:
        if code == 'B02':
            continue
        print(f"↪ Resampling {code} to match B02...")
        bands[code] = read_and_resample_band(
            band_paths[code],
            ref_shape=ref_shape,
            ref_transform=ref_transform,
            ref_crs=ref_crs
        )

    # ============================================================
    # --- Calculate Spectral Indices ---
    # ============================================================

    print("⚙ Calculating selected spectral indices...")

    indices = {
        "NDVI": calculate_index(bands['B08'], bands['B04']),
        "NDWI": calculate_index(bands['B03'], bands['B08']),
        "NDBI": calculate_index(bands['B11'], bands['B08']),
        "NDMI": calculate_index(bands['B08'], bands['B11']),
        "SAVI": ((bands['B08'] - bands['B04']) / (bands['B08'] + bands['B04'] + 0.5)) * 1.5,
        "BAI": 1.0 / ((0.1 - normalize(bands['B04']))**2 + (0.06 - normalize(bands['B08']))**2),
    }

    # ============================================================
    # --- Feature Masks ---
    # ============================================================

    print("🧪 Extracting feature masks...")

    masks = {
        "NDVI": extract_features(indices['NDVI'], threshold=0.3, mode='above'),
        "NDWI": extract_features(indices['NDWI'], threshold=0.1, mode='above'),
        "NDBI": extract_features(indices['NDBI'], threshold=0.1, mode='above'),
        "NDMI": extract_features(indices['NDMI'], threshold=0.2, mode='above'),
        "SAVI": extract_features(indices['SAVI'], threshold=0.25, mode='above'),
        "BAI": extract_features(indices['BAI'], threshold=0.1, mode='above'),
    }

    # ============================================================
    # --- Plotting ---
    # ============================================================

    print("📊 Plotting results per index...")

    mask_colors = {
        "NDVI": ('Greens', 'green', 'Vegetation'),
        "NDWI": ('Blues', 'blue', 'Water'),
        "NDBI": ('Reds', 'red', 'Built-up'),
        "NDMI": ('Purples', 'purple', 'Moisture'),
        "SAVI": ('YlGn', 'darkgreen', 'Soil/Vegetation'),
        "BAI": ('Oranges', 'orange', 'Burned Area'),
    }

    output_dir = os.path.join(safe_folder, "output")
    os.makedirs(output_dir, exist_ok=True)

    for name in indices:
        print(f"📈 Plotting {name}, RGB composite, and feature mask...")

        cmap_index, legend_color, legend_label = mask_colors[name]

        # --- Composite RGB (always True Color for this figure, with stretch) ---
        rgb = np.dstack([
            bands['B04'],  # Red
            bands['B03'],  # Green
            bands['B02']   # Blue
        ]).astype(np.float32)
        rgb = stretch_rgb(rgb)

        # Colormaps for index visualization
        index_cmaps = {
            "NDVI": "RdYlGn",
            "NDWI": "Blues",
            "NDBI": "OrRd",
            "NDMI": "BrBG",
            "SAVI": "YlGn",
            "BAI": "hot"
        }
        cmap_for_index = index_cmaps.get(name, "viridis")

        # --- Create figure with 3 subplots ---
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{name} - RGB Composite, Index, and Feature Mask", fontsize=16)

        # 1️⃣ RGB Composite
        axs[0].imshow(rgb)
        axs[0].set_title("True Color Composite (B04-B03-B02)")
        axs[0].axis("off")

        # 2️⃣ Index map
        im1 = axs[1].imshow(indices[name], cmap=cmap_for_index, vmin=-1, vmax=1)
        axs[1].set_title(f"{name} Index Map")
        axs[1].axis("off")
        cbar = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        cbar.set_label(f"{name} Value")

        # 3️⃣ Feature mask
        axs[2].imshow(masks[name], cmap=cmap_index)
        axs[2].set_title(f"{legend_label} Mask")
        axs[2].axis("off")
        legend_patch = Patch(color=legend_color, label=legend_label)
        axs[2].legend(handles=[legend_patch], loc="lower right")

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        if save_plot:
            output_path = os.path.join(output_dir, f"{name}_plot.png")
            plt.savefig(output_path, dpi=150)
            print(f"💾 {name} plot saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    # 🌈 Extra standalone RGB composites
    print("📸 Saving standalone RGB composites...")
    plot_rgb_composite(bands, "Natural Color", save_plot, output_dir)
    plot_rgb_composite(bands, "False Color", save_plot, output_dir)

    print(f"✅ All processing completed in {time.time() - t0:.2f} seconds")

# ============================================================
# --- Entry Point ---
# ============================================================

if __name__ == "__main__":
    print("📁 Paste the path to your Sentinel-2 .SAFE folder below.")
    safe_folder = input("📌 ").strip()

    if not os.path.exists(safe_folder):
        print("❌ Invalid folder path.")
    else:
        process_and_plot(safe_folder, save_plot=True)
