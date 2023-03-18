import pandas as pd
from pyrepseq.io import *
import pytest
import warnings


class TestStandardizeDataFrame:
    df_old = pd.DataFrame(
        [
            ['TRAJ23*01', 'CATQYF', 'TRAV26-1*01', 'TRBJ2-3*01', 'CASQYF', 'TRBV13*01', 'aaa', 'A1', 'B2M', 1],
            [None, 'CATQYF', None, None, 'CASQYF', None, None, None, None, 1],
            ['foobar', 'ATQY', 'TRAV26-1*01', 'TRBJ2-3*01', 'CASQYF', 'TRBV13*01', 'AAA', 'foobar', 'B2M', 1]
        ],
        columns=range(10)
    )
    df_standardized = pd.DataFrame(
        [
            ['TRAJ23', 'CATQYF', 'TRAV26-1', 'TRBJ2-3', 'CASQYF', 'TRBV13', 'AAA', 'HLA-A', 'B2M', 1],
            [None, 'CATQYF', None, None, 'CASQYF', None, None, None, None, 1],
            [None, 'CATQYF', 'TRAV26-1', 'TRBJ2-3', 'CASQYF', 'TRBV13', 'AAA', None, 'B2M', 1]
        ],
        columns=["TRAV", "CDR3A","TRAJ", "TRBV", "CDR3B", "TRBJ", "Epitope", "MHCA", "MHCB", "clonal_counts"]
    )


    def test_standardize_df(self):
        with pytest.warns(UserWarning, match='Failed to standardize'):
            result = standardize_dataframe(
                df_old=self.df_old,
                col_mapper={
                    i: self.df_standardized.columns[i] for i in range(10)
                }
            )

        assert result.equals(self.df_standardized)


    def test_no_standardization(self):
        result = standardize_dataframe(
            df_old=self.df_old,
            col_mapper={i:i for i in range(10)},
            standardize=False
        )

        assert result.equals(self.df_old)


    def test_suppress_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = standardize_dataframe(
                df_old=self.df_old,
                col_mapper={
                    i: self.df_standardized.columns[i] for i in range(10)
                },
                suppress_warnings=True
            )
        
        assert result.equals(self.df_standardized)
    
class TestSequenceExtract:
    a1ao7 = "KEVEQNSGPLSVPEGAIASLNCTYSDRGSQSFFWYRQYSGKSPELIMSIYSNGDKEDGRFTAQLNKASQYVSLLIRDSQPSDSATYLCAVTTDSWGKLQFGAGTQVVVTPDIQNPDPAVYQLRDSKSSDKSVCLFTDFDSQTNVSQSKDSDVYITDKTVLDMRSMDFKSNSAVAWSNKSDFACANAFNNSIIPEDTFFPSPESS"
    a1ao7_r = "-KEVEQNSGPLSVPEGAIASLNCTYSDRG------SQSFFWYRQYSGKSPELIMSIYS----NGDKED-----GRFTAQLNKASQYVSLLIRDSQPSDSATYLCAVTTDS--WGKLQFGAGTQVVVTP"
    b1ao7 = "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASRPGLAGGRPEQYFGPGTRLTVTEDLKNVFPPEVAVFEPSEAEISHTQKATLVCLATGFYPDHVELSWWVNGKEVHSGVSTDPQPLKEQPALNDSRYALSSRLRVSATFWQNPRNHFRCQVQFYGLSENDEWTQDRAKPVTQIVSAEAWGRAD"
    b1ao7_r = "NAGVTQTPKFQVLKTGQSMTLQCAQDMNH-------EYMSWYRQDPGMGLRLIHYSVG----AGITDQGEVP-NGYNVSRS-TTEDFPLRLLSAAPSQTSVYFCASRPGLAGGRPEQYFGPGTRLTVT"
    
    def test_get_v_region_seq(self):
        a = get_vregion_seq(self.a1ao7, gaps='yes')
        a1 = get_vregion_seq(self.a1ao7)

        assert a == self.a1ao7_r
        assert a1 == self.a1ao7_r.replace("-", "")

        b = get_vregion_seq(self.b1ao7, gaps = 'yes')
        b1 = get_vregion_seq(self.b1ao7)

        assert b == self.b1ao7_r
        assert b1 == self.b1ao7_r.replace("-", "")

    def test_get_cdrs(self):
        assert get_cdr1_seq(self.a1ao7, gaps='yes') == 'DRG------SQS'
        assert get_cdr2_seq(self.a1ao7, gaps='yes') == 'IYS----NGD'
        assert get_cdr3_seq(self.a1ao7, gaps='yes') == 'CAVTTDS--WGKLQF'
        assert get_cdr1_seq(self.a1ao7) == 'DRGSQS'
        assert get_cdr2_seq(self.a1ao7) == 'IYSNGD'
        assert get_cdr3_seq(self.a1ao7) == 'CAVTTDSWGKLQF'