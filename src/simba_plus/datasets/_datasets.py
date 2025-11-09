import urllib.request
from tqdm import tqdm
import os

from .._settings import settings
from ..readwrite import read_h5ad


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, desc=None):
    if desc is None:
        desc = url.split("/")[-1]
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
        try:
            urllib.request.urlretrieve(
                url, filename=output_path, reporthook=t.update_to
            )
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            raise e


def rna_10xpmbc3k():
    """10X human peripheral blood mononuclear cells (PBMCs) scRNA-seq data

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = "https://www.dropbox.com/s/087wuliddmbp3oe/rna_seq.h5ad?dl=1"
    filename = "rna_10xpmbc3k.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath = os.path.join(filepath, filename)
    if not os.path.exists(fullpath):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f"Downloaded to {filepath}.")
    adata = read_h5ad(fullpath)
    return adata


def rna_han2018():
    """single-cell microwell-seq mouse cell atlas data

    ref: Han, X. et al. Mapping the mouse cell atlas by microwell-seq.
    Cell 172, 1091-1107. e1017 (2018).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = "https://www.dropbox.com/s/nxbszjbir44g99n/rna_seq_mi.h5ad?dl=1"
    filename = "rna_han2018.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath = os.path.join(filepath, filename)
    if not os.path.exists(fullpath):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f"Downloaded to {filepath}.")
    adata = read_h5ad(fullpath)
    return adata


def rna_tmc2018():
    """single-cell Smart-Seq2 mouse cell atlas data

    ref: Tabula Muris Consortium. Single-cell transcriptomics of 20 mouse
    organs creates a Tabula Muris. Nature 562, 367-372 (2018).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = "https://www.dropbox.com/s/rnpyp6vfpuiptkz/rna_seq_sm.h5ad?dl=1"
    filename = "rna_tmc2018.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath = os.path.join(filepath, filename)
    if not os.path.exists(fullpath):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f"Downloaded to {filepath}.")
    adata = read_h5ad(fullpath)
    return adata


def rna_baron2016():
    """single-cell RNA-seq human pancreas data

    ref: Baron, M. et al. A single-cell transcriptomic map of the human and
    mouse pancreas reveals inter-and intra-cell population structure. Cell
    systems 3, 346-360. e344 (2016)

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = "https://www.dropbox.com/s/bvziclu6d3fdzow/rna_seq_baron.h5ad?dl=1"
    filename = "rna_baron2016.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath = os.path.join(filepath, filename)
    if not os.path.exists(fullpath):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f"Downloaded to {filepath}.")
    adata = read_h5ad(fullpath)
    return adata


def rna_muraro2016():
    """single-cell RNA-seq human pancreas data

    ref: Muraro, M.J. et al. A single-cell transcriptome atlas of the
    human pancreas.Cell systems 3, 385-394. e383 (2016).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = "https://www.dropbox.com/s/ginc9rbo4qmobwx/rna_seq_muraro.h5ad?dl=1"
    filename = "rna_muraro2016.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath = os.path.join(filepath, filename)
    if not os.path.exists(fullpath):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f"Downloaded to {filepath}.")
    adata = read_h5ad(fullpath)
    return adata


def rna_segerstolpe2016():
    """single-cell RNA-seq human pancreas data

    ref: Segerstolpe, Å. et al. Single-cell transcriptome profiling of human
    pancreatic islets in health and type 2 diabetes.
    Cell metabolism 24, 593-607 (2016).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = "https://www.dropbox.com/s/qomnf4860jwm9pd/rna_seq_segerstolpe.h5ad?dl=1"
    filename = "rna_segerstolpe2016.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath = os.path.join(filepath, filename)
    if not os.path.exists(fullpath):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f"Downloaded to {filepath}.")
    adata = read_h5ad(fullpath)
    return adata


def rna_wang2016():
    """single-cell RNA-seq human pancreas data

    ref: Wang, Y.J. et al. Single-cell transcriptomics of the human endocrine
    pancreas. Diabetes 65, 3028-3038 (2016).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = "https://www.dropbox.com/s/9tv44nugwpx9t4c/rna_seq_wang.h5ad?dl=1"
    filename = "rna_wang2016.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath = os.path.join(filepath, filename)
    if not os.path.exists(fullpath):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f"Downloaded to {filepath}.")
    adata = read_h5ad(fullpath)
    return adata


def rna_xin2016():
    """single-cell RNA-seq human pancreas data

    ref: Xin, Y. et al. RNA sequencing of single human islet cells reveals
    type 2 diabetes genes. Cell metabolism 24, 608-615 (2016).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = "https://www.dropbox.com/s/j483i47mxty6rzo/rna_seq_xin.h5ad?dl=1"
    filename = "rna_xin2016.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath = os.path.join(filepath, filename)
    if not os.path.exists(fullpath):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f"Downloaded to {filepath}.")
    adata = read_h5ad(fullpath)
    return adata


def atac_buenrostro2018():
    """single cell ATAC-seq human blood data

    ref: Buenrostro, J.D. et al. Integrated Single-Cell Analysis Maps the
    Continuous RegulatoryLandscape of Human Hematopoietic Differentiation.
    Cell 173, 1535-1548 e1516 (2018).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = "https://www.dropbox.com/s/7hxjqgdxtbna1tm/atac_seq.h5ad?dl=1"
    filename = "atac_buenrostro2018.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath = os.path.join(filepath, filename)
    if not os.path.exists(fullpath):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f"Downloaded to {filepath}.")
    adata = read_h5ad(fullpath)
    return adata


def atac_10xpbmc5k():
    """10X human peripheral blood mononuclear cells (PBMCs) scATAC-seq data

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = "https://www.dropbox.com/s/xa8u7rlskc5h7iv/atac_seq.h5ad?dl=1"
    filename = "atac_10xpbmc5k.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath = os.path.join(filepath, filename)
    if not os.path.exists(fullpath):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f"Downloaded to {filepath}.")
    adata = read_h5ad(fullpath)
    return adata


def atac_cusanovich2018_subset():
    """downsampled sci-ATAC-seq mouse tissue data

    ref: Cusanovich, D.A. et al. A Single-Cell Atlas of In Vivo Mammalian
    Chromatin Accessibility. Cell 174, 1309-1324 e1318 (2018).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = "https://www.dropbox.com/s/e8iqwm93m33i5wt/atac_seq.h5ad?dl=1"
    filename = "atac_cusanovich2018_subset.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath = os.path.join(filepath, filename)
    if not os.path.exists(fullpath):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f"Downloaded to {filepath}.")
    adata = read_h5ad(fullpath)
    return adata


def atac_chen2019():
    """simulated scATAC-seq bone marrow data with a noise level of 0.4
    and a coverage of 2500 fragments

    ref: Chen, H. et al. Assessment of computational methods for the analysis
    of single-cell ATAC-seq data. Genome Biology 20, 241 (2019).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = "https://www.dropbox.com/s/fthhh3mz5b39d4y/atac_seq.h5ad?dl=1"
    filename = "atac_chen2019.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath = os.path.join(filepath, filename)
    if not os.path.exists(fullpath):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url, fullpath, desc=filename)
        print(f"Downloaded to {filepath}.")
    adata = read_h5ad(fullpath)
    return adata


def multiome_ma2020_fig4():
    """single cell multiome mouse skin data (SHARE-seq)

    ref: Ma, S. et al. Chromatin Potential Identified by Shared Single-Cell
    Profiling of RNA and Chromatin. Cell (2020).

    Returns
    -------
    dict_adata: `dict`
        A dictionary of anndata objects
    """
    url_rna = "https://www.dropbox.com/s/gmmf77l8kzle6o7/rna_seq_fig4.h5ad?dl=1"
    url_atac = "https://www.dropbox.com/s/ts0v2y2m5fcumcb/atac_seq_fig4.h5ad?dl=1"
    filename_rna = "multiome_ma2020_fig4_rna.h5ad"
    filename_atac = "multiome_ma2020_fig4_atac.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath_rna = os.path.join(filepath, filename_rna)
    fullpath_atac = os.path.join(filepath, filename_atac)

    if not os.path.exists(fullpath_rna):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url_rna, fullpath_rna, desc=filename_rna)
        print(f"Downloaded to {filepath}.")
    if not os.path.exists(fullpath_atac):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url_atac, fullpath_atac, desc=filename_atac)
        print(f"Downloaded to {filepath}.")
    adata_rna = read_h5ad(fullpath_rna)
    adata_atac = read_h5ad(fullpath_atac)
    dict_adata = {"rna": adata_rna, "atac": adata_atac}
    return dict_adata


def multiome_chen2019():
    """single cell multiome neonatal mouse cerebral cortex data (SNARE-seq)

    ref: Chen, S., Lake, B.B. & Zhang, K. High-throughput sequencing of the
    transcriptome and chromatin accessibility in the same cell.
    Nat Biotechnol (2019).

    Returns
    -------
    dict_adata: `dict`
        A dictionary of anndata objects
    """
    url_rna = "https://www.dropbox.com/s/b1bbcs500q0pigt/rna_seq.h5ad?dl=1"
    url_atac = "https://www.dropbox.com/s/ljepkfber68pdvc/atac_seq.h5ad?dl=1"
    filename_rna = "multiome_chen2019_rna.h5ad"
    filename_atac = "multiome_chen2019_atac.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath_rna = os.path.join(filepath, filename_rna)
    fullpath_atac = os.path.join(filepath, filename_atac)

    if not os.path.exists(fullpath_rna):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url_rna, fullpath_rna, desc=filename_rna)
        print(f"Downloaded to {filepath}.")
    if not os.path.exists(fullpath_atac):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url_atac, fullpath_atac, desc=filename_atac)
        print(f"Downloaded to {filepath}.")
    adata_rna = read_h5ad(fullpath_rna)
    adata_atac = read_h5ad(fullpath_atac)
    dict_adata = {"rna": adata_rna, "atac": adata_atac}
    return dict_adata


def multiome_10xpbmc10k():
    """single cell 10X human peripheral blood mononuclear cells (PBMCs)
    multiome data

    Returns
    -------
    dict_adata: `dict`
        A dictionary of anndata objects
    """
    url_rna = "https://www.dropbox.com/s/zwlim6vljnbfp43/rna_seq.h5ad?dl=1"
    url_atac = "https://www.dropbox.com/s/163msz0k9hkfrt7/atac_seq.h5ad?dl=1"
    filename_rna = "multiome_10xpbmc10k_rna.h5ad"
    filename_atac = "multiome_10xpbmc10k_atac.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath_rna = os.path.join(filepath, filename_rna)
    fullpath_atac = os.path.join(filepath, filename_atac)

    if not os.path.exists(fullpath_rna):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url_rna, fullpath_rna, desc=filename_rna)
        print(f"Downloaded to {filepath}.")
    if not os.path.exists(fullpath_atac):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url_atac, fullpath_atac, desc=filename_atac)
        print(f"Downloaded to {filepath}.")
    adata_rna = read_h5ad(fullpath_rna)
    adata_atac = read_h5ad(fullpath_atac)
    dict_adata = {"rna": adata_rna, "atac": adata_atac}
    return dict_adata


def multiome_bmmc():
    """single cell 10X human peripheral blood mononuclear cells (PBMCs)
    multiome data

    Returns
    -------
    dict_adata: `dict`
        A dictionary of anndata objects
    """
    url_rna = "https://www.dropbox.com/s/zwlim6vljnbfp43/rna_seq.h5ad?dl=1"
    url_atac = "https://www.dropbox.com/s/163msz0k9hkfrt7/atac_seq.h5ad?dl=1"
    filename_rna = "multiome_10xpbmc10k_rna.h5ad"
    filename_atac = "multiome_10xpbmc10k_atac.h5ad"
    filepath = os.path.join(settings.workdir, "data")
    fullpath_rna = os.path.join(filepath, filename_rna)
    fullpath_atac = os.path.join(filepath, filename_atac)

    if not os.path.exists(fullpath_rna):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url_rna, fullpath_rna, desc=filename_rna)
        print(f"Downloaded to {filepath}.")
    if not os.path.exists(fullpath_atac):
        print("Downloading data ...")
        os.makedirs(filepath, exist_ok=True)
        download_url(url_atac, fullpath_atac, desc=filename_atac)
        print(f"Downloaded to {filepath}.")
    adata_rna = read_h5ad(fullpath_rna)
    adata_atac = read_h5ad(fullpath_atac)
    dict_adata = {"rna": adata_rna, "atac": adata_atac}
    return dict_adata


def extract_tarfile(tar_path, extract_path=None, remove_original=True):
    import tarfile

    if extract_path is None:
        extract_path = os.path.dirname(tar_path)
    os.makedirs(extract_path, exist_ok=True)
    try:
        # Open the .tgz file in read mode ('r:gz' for gzipped tar files)
        with tarfile.open(tar_path, "r:gz") as tar:
            # Extract all contents to the specified destination directory
            tar.extractall(extract_path)
            print(f"Successfully extracted '{tar_path}' to '{extract_path}'")
    except tarfile.ReadError as e:
        print(f"Error opening or reading tar file: {e}")
    except FileNotFoundError:
        print(f"Error: The file '{tar_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    if remove_original:
        os.remove(tar_path)


def heritability(logger=None):
    """placeholder for heritability dataset loader

    Returns
    -------
    None
    """

    def _log(msg):
        if logger is None:
            print(msg)
        else:
            logger.info(msg)

    url_baseline = "https://zenodo.org/records/10515792/files/1000G_Phase3_baselineLD_v2.2_ldscores.tgz?download=1"
    url_weights_hm3 = "https://zenodo.org/records/10515792/files/1000G_Phase3_weights_hm3_no_MHC.tgz?download=1"
    url_frqfile = (
        "https://zenodo.org/records/10515792/files/1000G_Phase3_frq.tgz?download=1"
    )
    url_plink = "https://zenodo.org/records/10515792/files/1000G_Phase3_plinkfiles.tgz?download=1"
    url_snplist = (
        "https://zenodo.org/records/10515792/files/hm3_no_MHC.list.txt?download=1"
    )

    filepath_prefix = os.path.join(
        os.path.dirname(__file__), "../../../data/ldsc_data/"
    )
    fullpath_baseline = os.path.join(
        filepath_prefix, "1000G_Phase3_baselineLD_v2.2_ldscores.tgz"
    )
    fullpath_weights_hm3 = os.path.join(
        filepath_prefix, "1000G_Phase3_weights_hm3_no_MHC.tgz"
    )
    fullpath_frqfile = os.path.join(filepath_prefix, "1000G_Phase3_frq.tgz")
    fullpath_plink = os.path.join(filepath_prefix, "1000G_Phase3_plinkfiles.tgz")
    fullpath_snplist = os.path.join(filepath_prefix, "hm3_no_MHC.list.txt")

    if not os.path.exists(filepath_prefix):
        os.makedirs(filepath_prefix)

    if not os.path.exists(fullpath_baseline.split(".tgz")[0]):
        if not os.path.exists(fullpath_baseline):
            _log(f"Downloading baseline LD scores to {fullpath_baseline}...")
            download_url(
                url_baseline, fullpath_baseline, desc="baselineLD_v2.2_ldscores.tgz"
            )
        extract_tarfile(fullpath_baseline)
        _log(f"Downloaded to {fullpath_baseline}.")

    if not os.path.exists(fullpath_weights_hm3.split(".tgz")[0]):
        if not os.path.exists(fullpath_weights_hm3):
            _log(f"Downloading weights hm3 to {fullpath_weights_hm3}...")
            download_url(
                url_weights_hm3, fullpath_weights_hm3, desc="weights_hm3_no_MHC.tgz"
            )
        extract_tarfile(fullpath_weights_hm3)
        _log(f"Downloaded to {fullpath_weights_hm3}.")

    if not os.path.exists(fullpath_frqfile.split(".tgz")[0]):
        if not os.path.exists(fullpath_frqfile):
            _log(f"Downloading frq file to {fullpath_frqfile}...")
            download_url(url_frqfile, fullpath_frqfile, desc="frq.tgz")
        extract_tarfile(fullpath_frqfile)
        _log(f"Downloaded to {fullpath_frqfile}.")

    bedfile_dir = fullpath_plink.split(".tgz")[0]
    bedfile_allchrom = os.path.join(bedfile_dir, "ref.txt")
    if not os.path.exists(bedfile_allchrom):
        if not os.path.exists(fullpath_plink.split(".tgz")[0]):
            if not os.path.exists(fullpath_plink):
                _log(f"Downloading plink file to {fullpath_plink}...")
                download_url(url_plink, fullpath_plink, desc="plink.tgz")
            extract_tarfile(fullpath_plink)
            _log(f"Downloaded to {fullpath_plink}.")

        for i in range(1, 23):
            bimfile_path = os.path.join(
                bedfile_dir, f"1000G_EUR_Phase3_plink/1000G.EUR.QC.{i}.bim"
            )
            if not os.path.exists(bimfile_path):
                _log(f"Error: missing PLINK bed file {bimfile_path}")
            else:
                os.system(
                    "awk -v OFS='\\t' '{print $2, $1, $4}' "
                    + bimfile_path
                    + " >> "
                    + bedfile_allchrom
                )

    if not os.path.exists(fullpath_snplist):
        _log(f"Downloading SNP list to {fullpath_snplist}...")
        download_url(url_snplist, fullpath_snplist, desc="hm3_no_MHC.list.txt")
        _log(f"Downloaded to {fullpath_snplist}.")


def sumstats(logger=None):
    def _log(msg):
        if logger is None:
            print(msg)
        else:
            logger.info(msg)

    sumstats_url = "https://figshare.com/ndownloader/files/34300928"
    filename = "sumstats.txt"
    filepath_prefix = os.path.join(os.path.dirname(__file__), "../../../data/sumstats/")
    fullpath_sumstats = os.path.join(filepath_prefix, filename)

    if not os.path.exists(fullpath_sumstats):
        _log(f"Downloading sumstats to {fullpath_sumstats}...")
        os.makedirs(filepath_prefix, exist_ok=True)
        download_url(sumstats_url, fullpath_sumstats, desc=filename)
        _log(f"Downloaded to {fullpath_sumstats}.")
    else:
        _log(f"Sumstats already present at {fullpath_sumstats}, skipping download.")
