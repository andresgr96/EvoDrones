"""Module to generate tracks for the simulation"""

import xml.etree.ElementTree as ET
import math
from xml.dom import minidom
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.affinity import rotate, translate


def prettify(elem: str) -> str:
    """
    Return a pretty-printed XML string for the Element.
    Args:
        elem (str): xml document as string
    Returns:
        str: xml document as string prettified (not just one line but with newlines, etc.)"""
    re_parsed = minidom.parseString(elem)
    return re_parsed.toprettyxml(indent="  ")


def create_urdf_segment(
        segment_id,
        start_point,
        end_point, width,
        track_color,
        height_offset
    ) -> tuple[ET.Element]:
    """Create a URDF segment in XML format.

    Args:
        segment_id (int): id of the segment (has to be unique). Used as name in the xml document
                          for the segment
        start_point (tuple): (x, y, z) starting point of the segment (absolute coordinate)
        end_point (tuple): (x, y, z) end point of the segment 
        width (float): width of the segment
        track_color (str): 'R G B' color of the segment. Should be string with three values between
                            0 and 1
        height_offset (float): height offset of the track, e.g. the height of the track above ground

    Returns:
        tuple[ET.Element]: (link, joint), the link is the actual segment, the joint connects the
                           segment with the next segment
    """
    link = ET.Element("link", name=f"segment_{segment_id}")

    visual = ET.SubElement(link, "visual")
    geometry = ET.SubElement(visual, "geometry")

    overlap = width
    # Calculate segment length and angle for rotation
    length = math.sqrt(
        (end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2
    ) + overlap
    angle = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])

    # Define box geometry
    box = ET.SubElement(geometry, "box")
    box.set("size", f"{length} {width} {height_offset}")

    # Set the origin (position and rotation) of the segment
    origin = ET.SubElement(visual, "origin")
    x = (start_point[0] + end_point[0]) / 2
    y = (start_point[1] + end_point[1]) / 2
    origin.set("xyz", f"{x} {y} 0.0")
    origin.set("rpy", f"0 0 {angle}")  # Rotate around the z-axis (x, y)-plane

    # Define material and color
    material = ET.SubElement(visual, "material", name="track_color")
    color = ET.SubElement(material, "color")
    color.set("rgba", f"{track_color} 1")

    joint = ET.Element("joint", name=f"joint_{segment_id}", type="fixed")
    parent_link = f"segment_{segment_id - 1}" if segment_id > 0 else "root_link"
    ET.SubElement(joint, "parent", link=parent_link)
    ET.SubElement(joint, "child", link=f"segment_{segment_id}")

    return link, joint


def create_landing_point(
        last_segment_id: int,
        center_point: tuple,
        radius: float,
        track_color: str,
        height_offset: float
    ) -> tuple[ET.Element]:
    """Create the landing point (circle at the end of the track). Very similar to the segment 
    creation.

    Args:
        last_segment_id (int): id of the previous (last) segment
        center_point (tuple): center point of the circle. Should be the end of the last segment
        radius (float): radius of the circle
        track_color (str): 'R G B' color of the circle. Should be string with three values between
                            0 and 1
        height_offset (float): height offset of the track, e.g. the height of the track above ground

    Returns:
        _type_: _description_
    """
    link = ET.Element("link", name="goal")

    visual = ET.SubElement(link, "visual")
    geometry = ET.SubElement(visual, "geometry")
    # Define box geometry
    cylinder = ET.SubElement(geometry, "cylinder", radius=str(radius), length=f"{height_offset}")
    # Set the origin (position and rotation) of the circle
    origin = ET.SubElement(visual, "origin", xyz=f"{center_point[0]} {center_point[1]} 0.0")
    # Define material and color
    material = ET.SubElement(visual, "material", name="track_color")
    color = ET.SubElement(material, "color", rgba=f"{track_color} 1")

    joint = ET.Element("joint", name="joint_goal", type="fixed")
    parent_link = f"segment_{last_segment_id}" if last_segment_id > 0 else "root_link"
    ET.SubElement(joint, "parent", link=parent_link)
    ET.SubElement(joint, "child", link="goal")

    return link, joint


def generate_track_urdf(track_points: list[tuple], width: float, color: str, height_offset=0.002):
    """ Generate a URDF file for a given track.

    Args:
        track_points (list[tuple]): Waypoint of the track at which the segments change
        width (float): width of the track
        color (tuple): (R, G, B) color of the track. Values should be between 0 and 1 (inclusive)
        height_offset (float, optional): Height of the track. Defaults to 0.002

    Returns:
        ET.Element: The track as Element object
    """
    urdf = ET.Element("robot", name="track")
    color = str(color)
    # Create a root link
    root_link = ET.Element("link", name="root_link")
    urdf.append(root_link)
    # create segments
    for i in range(len(track_points) - 1):
        segment, joint = create_urdf_segment(
            segment_id=i,
            start_point= track_points[i],
            end_point=track_points[i+1],
            width=width,
            track_color=color,
            height_offset=height_offset
        )
        urdf.append(segment)
        urdf.append(joint)
    # create landing spot
    circle, joint = create_landing_point(
        len(track_points)-2,
        track_points[-1],
        width,
        color,
        height_offset
    )
    urdf.append(circle)
    urdf.append(joint)

    return urdf


def save_track(urdf_element: ET.Element, track_path: str) -> None:
    """Saves the track to the file system.

    Args:
        urdf_element (ET.Element): track to be saved
        track_path (str): path where the track should be saved
    """
    urdf_string = ET.tostring(urdf_element, encoding='utf-8')
    with open(track_path, "w", encoding="utf-8") as file:
        file.write(prettify(urdf_string))


def load_root(track_path: str) -> ET.Element:
    """Loads the track from the file system and return the root of the xml file

    Args:
        track_path (str): path where the track is saved
    """
    # Parse the XML file
    tree = ET.parse(track_path)
    return tree.getroot()


def parse_floats(string) -> list:
    """Helper function to parse a string of space-separated floats."""
    return list(map(float, string.split()))


def calculate_rotated_polygon(size, xy, yaw) -> Polygon:
    """
    Calculate the bounds of the rotated box in 2D.

    Args:
        size (list): [length, width] Size of the box
        xy (list): [x, y] Origin position of the box
        yaw (float): Yaw rotation of the box (in radians)
    Returns:
        Polygon: shapely Polygon of the rotated box
    """
    # Create a box centered at origin
    dx, dy = size[0] / 2, size[1] / 2
    box = Polygon([(-dx, -dy), (-dx, dy), (dx, dy), (dx, -dy)])
    # Rotate and translate the box to the correct position
    rotated_box = rotate(box, yaw, use_radians=True, origin=(0, 0))
    final_box = translate(rotated_box, *xy)

    return final_box


def load_segments(root: ET.Element) -> dict:
    """
    Method to load the segments from a file into a dict. Can be used for fitness evaluation.
    The dict contains segment_id keys and dict as values. The values contain 'box' which is
    the Polygon to determine if a point is within a segment. as a second key 'forward' is available
    which is the unit norm of the direction of the segment (notion of forward).

    Args:
        tree (ET.Element): ElementTree object representing a track root

    Returns:
        dict: Dictionary with information about the track
    """
    # Dictionary to hold segment names and their total coordinates
    segment_info = {}

    # Iterate over all 'link' elements
    for link in root.findall('link'):
        name = link.get('name')
        if not name:
            continue
        if name == "goal":
            cylinder = link.find('.//cylinder')
            origin = link.find('.//origin')
            if cylinder is not None and origin is not None:
                # Get the radius of the cylinder
                radius = float(cylinder.get('radius'))
                xy = parse_floats(origin.get('xyz'))[0:2]
                circle = Point(*xy).buffer(radius)
                segment_info[name] = {
                    "shape": circle
                }
        # Check if the link name starts with 'segment'
        elif name.startswith('segment'):
            box = link.find('.//box')
            origin = link.find('.//origin')
            if box is not None and origin is not None:
                # Extract size and origin
                size = parse_floats(box.get('size'))
                xy = parse_floats(origin.get('xyz'))[0:2]
                _, _, yaw = parse_floats(origin.get('rpy'))  # return: roll, pitch, yaw

                rotated_box = calculate_rotated_polygon(size, xy, yaw)
                # 2D forward direction
                unit_norm = np.array([np.cos(yaw), np.sin(yaw), 0])

                segment_info[name] = {
                    "shape": rotated_box,
                    "forward": unit_norm.tolist()
                }
    return segment_info


if __name__ == '__main__':
    # Example usage
    track_path = "/home/m/Desktop/Projects/gym-pybullet-drones/gym_pybullet_drones/assets/track.urdf"
    track_points = [(0, 0), (-1.5, 0), (-1.5, -1.5), (0, -1), (-0.5, -0.5)]  # Define your track points here
    path_width = 0.1
    path_color = (1, 0, 0)  # Color of the path (RGB, between 0 and 1)

    urdf_content = generate_track_urdf(track_points, path_width, path_color, height_offset=0.01)
    segments = load_segments(urdf_content)  # <- use the get dict containing box coordinates


    # Example usage of checking containment
    point = Point((-0.25, -0.75))
    print(segments["segment_3"]["shape"])
    print(segments["segment_3"]["shape"].contains(point))
    print(segments["goal"]["shape"].area)


    """
    Use this to add the track (replace the path!).

    p.loadURDF(
        track_file_path,
        useFixedBase=1,
        basePosition=[0, 0, 0],
        physicsClientId=self.CLIENT
    )
    """