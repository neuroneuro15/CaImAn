import os

try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen


demo_files = {'Sue_2x_3000_40_-46.tif': 'https://www.dropbox.com/s/09z974vkeg3t5gn/Sue_2x_3000_40_-46.tif?dl=1',
                 'demoMovieJ.tif': 'https://www.dropbox.com/s/8j1cnqubye3asmu/demoMovieJ.tif?dl=1',
                 'demo_behavior.h5': 'https://www.dropbox.com/s/53jmhc9sok35o82/movie_behavior.h5?dl=1',
                 'Tolias_mesoscope_1.hdf5': 'https://www.dropbox.com/s/t1yt35u0x72py6r/Tolias_mesoscope_1.hdf5?dl=1',
                 'Tolias_mesoscope_2.hdf5': 'https://www.dropbox.com/s/i233b485uxq8wn6/Tolias_mesoscope_2.hdf5?dl=1',
              'Tolias_mesoscope_3.hdf5': 'https://www.dropbox.com/s/4fxiqnbg8fovnzt/Tolias_mesoscope_3.hdf5?dl=1',
              'data_endoscope.tif':'https://www.dropbox.com/s/dcwgwqiwpaz4qgc/data_endoscope.tif?dl=1'}

def download_demo(name='Sue_2x_3000_40_-46.tif', save_folder='', base_folder='./example_movies'):
    """download a file from the file list with the url of its location


    using urllib, you can add you own name and location in this global parameter

        Parameters:
        -----------

        name: str
            the path of the file correspondong to a file in the filelist (''Sue_2x_3000_40_-46.tif' or 'demoMovieJ.tif')

        save_folder: str
            folder inside ./example_movies to which the files will be saved. Will be created if it doesn't exist

    Raise:
    ---------
        WrongFolder Exception


    """
    url = demo_files[name]

    savename = os.path.join(base_folder, save_folder, name)
    if os.path.exists(savename):
        print("File already downloaded. Skipping...")
        return

    if not os.path.isdir(os.path.dirname(savename)):
        os.makedirs(os.path.dirname(savename))

    print("downloading " + name + "with urllib")
    with open(savename, "wb") as code:
        code.write(urlopen(url).read())
