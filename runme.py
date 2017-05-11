import car_world
import os


if __name__ == '__main__':
    car = car_world.CarWorld()
    input_f = os.path.join("test_vid", "project_video.mp4")
    output_f = os.path.join(car.results_dir, "project_video_output.mp4")

    car.process_video(input_f, output_f)