import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import imageio
import os
import numpy as np
from io import BytesIO

# Load Data: Read the Excel File into a pandas DataFrame
df = pd.read_excel('ride_hailing.xlsx')

# Basic Processing:
# Create a new column called Status. If the reservation_id column has any text or number in it, 
# the status should be 'occupied'. otherwise it should be 'vacant'
df['Status'] = df['reservation_id'].apply(
    lambda x: 'occupied' if pd.notna(x) and str(x).strip() != '' else 'vacant'
)

# Make sure the current_time column is understood by the program as a date and time
df['current_time'] = pd.to_datetime(df['current_time'])

# Get all 24 slot positions (coordinates are consistent across timestamps)
slot_positions = df.dropna(subset=['x', 'y']).groupby('slot_id')[['x', 'y']].first().reset_index()

# Get all unique timestamps and sort them
unique_timestamps = sorted(df['current_time'].unique())
print(f"Found {len(unique_timestamps)} unique timestamps")

# Load background image and flip it vertically
bg_image = Image.open('map.png')
img_width, img_height = bg_image.size
# Flip the image vertically so it displays correctly
bg_image = bg_image.transpose(Image.FLIP_TOP_BOTTOM)

# Transform y-coordinates to match the flipped image
slot_positions['plot_y'] = img_height - slot_positions['y']

# Generate figure and axis (will be reused for each frame)
fig, ax = plt.subplots(figsize=(16, 12))

# Display background image (same for all frames)
ax.imshow(bg_image, extent=[0, img_width, 0, img_height],
          aspect='equal', zorder=0, origin='upper')

# Set axis limits - invert y-axis to match the flipped image and coordinates
ax.set_xlim(0, img_width)
ax.set_ylim(img_height, 0)  # Inverted: y=0 at top

# Remove white background and grid lines
ax.set_facecolor('none')
ax.grid(False)

# Hide x and y axis numbers and ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

# Remove axis spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Add legend (same for all frames)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Vacant', 
           markerfacecolor='green', markersize=10, markeredgecolor='black', markeredgewidth=0.5),
    Line2D([0], [0], marker='o', color='w', label='Nearest Vacant', 
           markerfacecolor='orange', markersize=10, markeredgecolor='black', markeredgewidth=0.5),
    Line2D([0], [0], marker='o', color='w', label='Entering Vehicle', 
           markerfacecolor='cyan', markersize=10, markeredgecolor='black', markeredgewidth=0.5),
    Line2D([0], [0], color='yellow', linestyle='-', linewidth=2.5, label='Path to Spot')
]
legend_obj = ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

# Store references to dynamic elements that will be updated
title_text_obj = ax.text(0.5, 1.02, '', transform=ax.transAxes, 
                         fontsize=16, fontweight='bold', ha='center', va='bottom')

def load_plate_image(plate_number, zoom=0.15):
    """Load and prepare a license plate image for annotation."""
    plate_path = f'Plates/{plate_number}.png'
    if os.path.exists(plate_path):
        img = Image.open(plate_path)
        return OffsetImage(img, zoom=zoom)
    return None

def animate_frame(frame_num):
    """Function to draw each frame of the animation."""
    # Clear scatter plots (collections)
    while ax.collections:
        ax.collections[0].remove()
    
    # Remove all annotation artists (AnnotationBbox objects are added as artists)
    # Title text is in ax.texts, not ax.artists, so safe to clear all artists
    while ax.artists:
        ax.artists[0].remove()

    # Clear lines (paths) from previous frame
    while ax.lines:
        ax.lines[0].remove()
    
    # Get current timestamp
    current_timestamp = unique_timestamps[frame_num]
    
    # Update title
    title_text_obj.set_text(f'Parking Status Visualization - {current_timestamp.strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Filter data for current timestamp
    timestamp_data = df[df['current_time'] == current_timestamp].copy()
    
    # Calculate status for each slot at this timestamp
    timestamp_data['Status'] = timestamp_data['reservation_id'].apply(
        lambda x: 'occupied' if pd.notna(x) and str(x).strip() != '' else 'vacant'
    )
    
    # Merge slot positions with status data
    slot_status = timestamp_data[['slot_id', 'Status', 'plate_number']].drop_duplicates('slot_id', keep='last')
    
    # Combine positions with status - all 24 slots should be shown
    plot_df = slot_positions.merge(slot_status, on='slot_id', how='left')
    plot_df['Status'] = plot_df['Status'].fillna('vacant')  # Default to vacant if no data
    
    # Separate occupied and vacant spots
    occupied_spots = plot_df[plot_df['Status'] == 'occupied'].copy()
    vacant_spots = plot_df[plot_df['Status'] == 'vacant'].copy()
    
    # Plot vacant spots
    if not vacant_spots.empty:
        # Calculate distance from top-left (0, 0) for each vacant spot
        # Using Euclidean distance: sqrt(x^2 + y^2)
        # Note: We use plot_y which corresponds to the visual coordinates
        vacant_spots['distance'] = np.sqrt(vacant_spots['x']**2 + vacant_spots['plot_y']**2)
        
        # Find the index of the nearest spot
        nearest_idx = vacant_spots['distance'].idxmin()
        
        # Split into nearest and others
        nearest_spot = vacant_spots.loc[[nearest_idx]]
        other_vacant = vacant_spots.drop(nearest_idx)
        
        # Plot other vacant spots as green dots
        if not other_vacant.empty:
            ax.scatter(other_vacant['x'], other_vacant['plot_y'], 
                      c='green', alpha=0.7, s=80, zorder=1,
                      edgecolors='black', linewidths=0.5)
            
        # Plot nearest vacant spot as orange dot
        ax.scatter(nearest_spot['x'], nearest_spot['plot_y'], 
                  c='orange', alpha=0.9, s=100, zorder=2,
                  edgecolors='black', linewidths=1.0)
                  
        # Draw path from entrance to nearest vacant spot
        # Entrance coordinates are defined below, but we use the values here
        ent_x, ent_y = 225, 50
        path_x = [ent_x, nearest_spot['x'].values[0]]
        path_y = [ent_y, nearest_spot['plot_y'].values[0]]
        
        ax.plot(path_x, path_y, color='yellow', linestyle='-', linewidth=2.5, zorder=1.5, alpha=0.9)
    
    # Plot occupied spots with license plate images
    for idx, row in occupied_spots.iterrows():
        plate_number = row['plate_number']
        if pd.notna(plate_number):
            plate_img = load_plate_image(plate_number)
            if plate_img is not None:
                # Create annotation box with license plate image
                ann = AnnotationBbox(plate_img, (row['x'], row['plot_y']), 
                                    frameon=False, zorder=2)
                ax.add_artist(ann)
        else:
            # If no plate number, use red dot as fallback
            ax.scatter(row['x'], row['plot_y'], c='red', alpha=0.7, s=80, 
                      zorder=1, edgecolors='black', linewidths=0.5)
    
    # Plot entering vehicle marker (static at top-left entrance)
    # Coordinates updated based on user feedback (blue dot, more to the right)
    entrance_x, entrance_y = 225, 50
    ax.scatter(entrance_x, entrance_y, c='cyan', alpha=1.0, s=150, 
              zorder=3, edgecolors='black', linewidths=1.5)
    
    return []

# Create animation frames - generate all frames
print("Generating animation frames...")
frames = []
for i in range(len(unique_timestamps)):
    animate_frame(i)
    fig.canvas.draw()
    
    # Save figure to a BytesIO buffer as PNG
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='white', transparent=False)
    buf.seek(0)
    
    # Read the PNG image and convert to numpy array
    frame_img = Image.open(buf)
    frame_array = np.array(frame_img)
    
    # Convert RGBA to RGB if needed (remove alpha channel)
    if frame_array.shape[2] == 4:
        frame_array = frame_array[:, :, :3]
    
    frames.append(frame_array)
    buf.close()
    
    if (i + 1) % 10 == 0:
        print(f"  Generated {i + 1}/{len(unique_timestamps)} frames...")

print(f"\nPreparing frames for 2-second display per minute...")
# Convert numpy arrays to PIL Images for better GIF control
# Frames are already in RGB format from earlier conversion
pil_frames = [Image.fromarray(frame_array, 'RGB') for frame_array in frames]

print(f"  Total frames: {len(pil_frames)}")
print(f"  Duration per frame: 2 seconds (2000 milliseconds)")

# Save as GIF using PIL's save method
# duration is in milliseconds, so 2000ms = 2 seconds per frame
print(f"\nSaving animation as 'parking_animation.gif'...")
pil_frames[0].save(
    'parking_animation.gif',
    save_all=True,
    append_images=pil_frames[1:],
    duration=2000,  # 2000 milliseconds = 2 seconds per frame
    loop=0  # Loop indefinitely
)

print("Animation saved successfully as 'parking_animation.gif'!")
plt.close()
