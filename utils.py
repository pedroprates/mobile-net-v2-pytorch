from six.moves import urllib
import os
import hashlib
from tqdm.auto import tqdm


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename=None, md5=None):
    root = os.path.expanduser(root)

    if not filename:
        filename = os.path.basename(url)

    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # Downloading the file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:','http:')
                print('Failed download. Trying https -> http instead.')
                print('Downloading ' + url + ' to ' + fpath)

                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())
