import numpy as np
import pdb
from PIL import Image

def frameGenerator( frame_size = [256, 256], num_of_frames = 2, move_set = ["right", "up"],
                    color_scale = 256, size_of_object = 4, movement_distance = 4):
    """
    Generates sets of frames that represent frame to frame movement for each frame set
    Adds object to each frame with location adjusted against previous frame movement

    Returns: set of frames representative of single movement

    Inputs:
        frame_size:
        num_of_frames:
        move_set:
        color_scale:
        size_of_object:
        movement_distance:
    Outputs: List of numpy n-d arrays. Zero index is initial frame. All subsequent frames follow each movement from move_set
    """
    # pdb.set_trace()

    # Evaluate inputs correct
    if num_of_frames != len(move_set):
        raise InputError('Number of frames generated doesn\'t match number of movements')
        return

    if len(frame_size) < 2:
        raise InputError('Frame Size is not at least 2d')
        return

    # Generate initial frame
    frame0 = np.random.randint(0,high=color_scale-1,size=frame_size)
    frame1 = frame0.copy()

    # Generate Random n-d location for square object
    num_of_dims = len(frame_size)
    object_idx = []
    for dim in range(num_of_dims):
        object_idx.append(np.random.randint(0,high= ( frame_size[dim] - size_of_object)))

    # Update frame0 with initial object placement
    object_x = object_y = object_z = 0
    object_val = np.random.randint(0,high=color_scale)

    if num_of_dims == 2:
        object_y = object_idx[0]
        object_x = object_idx[1]
        frame1[object_y:(object_y+size_of_object),object_x:(object_x+size_of_object)] = object_val

    elif num_of_dims == 3:
        object_y = object_idx[0]
        object_x = object_idx[1]
        object_z = object_idx[2]
        frame1[object_y:(object_y+size_of_object),object_x:(object_x+size_of_object),object_z:(object_z+size_of_object)] = object_val

    # Create frames with each new movement
    frame_set = [frame1]
    for frameIdx in range(num_of_frames):
        movement = move_set[frameIdx]

        copy = frame0.copy()

        # Adjust object origin
        if movement == 'left':
            object_x -= movement_distance
        elif movement == 'right':
            object_x += movement_distance
        elif movement == 'down':
            object_y += movement_distance
        elif movement == 'up':
            object_y -= movement_distance

        # Determine if movement causes object to move out of bounds, ignore this frame set
        if object_x < 0 or object_x+size_of_object >= frame_size[0] or object_y < 0 or object_y+size_of_object >= frame_size[1]:
            print("Got out of bounds")
            return []

        if num_of_dims == 2:
            copy[object_y:(object_y+size_of_object),object_x:(object_x+size_of_object)] = object_val

        elif num_of_dims == 3:
            copy[object_y:(object_y+size_of_object),object_x:(object_x+size_of_object),object_z:(object_z+size_of_object)] = object_val

        frame_set.append(copy)

    return frame_set
if __name__ == '__main__':
    frameGenerator()
