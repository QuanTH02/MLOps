from dotenv import load_dotenv
from crawl_update_mojo import *
from crawl_update_tmdb import *
from crawl_update_imdb import *
from crawl_update_critic_metascore import *
from crawl_update_themoviedb import *
from constant import DATA_FILE_NAME

load_dotenv()

if __name__ == '__main__':
    main_mojo(DATA_FILE_NAME)
    main_tmdb(DATA_FILE_NAME)
    main_imdb(DATA_FILE_NAME)
    main_critic_metascore(DATA_FILE_NAME)
    main_themoviedb(DATA_FILE_NAME)