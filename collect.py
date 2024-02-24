import multiprocessing as mp
import data_generator

if __name__ == '__main__':
    while True:
        subprocess = mp.Process(target=data_generator.run)
        subprocess.start()
        subprocess.join()
