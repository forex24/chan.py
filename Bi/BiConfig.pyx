# cython: language_level=3
from Common.CEnum cimport FX_CHECK_METHOD
from Common.ChanException cimport CChanException, ErrCode

cdef class CBiConfig:
    cdef:
        public str bi_algo
        public bint is_strict
        public FX_CHECK_METHOD bi_fx_check
        public bint gap_as_kl
        public bint bi_end_is_peak
        public bint bi_allow_sub_peak

    def __cinit__(self,
                  str bi_algo="normal",
                  bint is_strict=True,
                  str bi_fx_check="half",
                  bint gap_as_kl=True,
                  bint bi_end_is_peak=True,
                  bint bi_allow_sub_peak=True):
        self.bi_algo = bi_algo
        self.is_strict = is_strict
        
        if bi_fx_check == "strict":
            self.bi_fx_check = FX_CHECK_METHOD.STRICT
        elif bi_fx_check == "loss":
            self.bi_fx_check = FX_CHECK_METHOD.LOSS
        elif bi_fx_check == "half":
            self.bi_fx_check = FX_CHECK_METHOD.HALF
        elif bi_fx_check == 'totally':
            self.bi_fx_check = FX_CHECK_METHOD.TOTALLY
        else:
            raise CChanException(f"unknown bi_fx_check={bi_fx_check}", ErrCode.PARA_ERROR)

        self.gap_as_kl = gap_as_kl
        self.bi_end_is_peak = bi_end_is_peak
        self.bi_allow_sub_peak = bi_allow_sub_peak

    def __str__(self):
        return (f"CBiConfig(bi_algo={self.bi_algo}, is_strict={self.is_strict}, "
                f"bi_fx_check={self.bi_fx_check}, gap_as_kl={self.gap_as_kl}, "
                f"bi_end_is_peak={self.bi_end_is_peak}, bi_allow_sub_peak={self.bi_allow_sub_peak})")

    def __repr__(self):
        return self.__str__()

    cpdef CBiConfig copy(self):
        return CBiConfig(
            bi_algo=self.bi_algo,
            is_strict=self.is_strict,
            bi_fx_check=self.bi_fx_check.name.lower(),
            gap_as_kl=self.gap_as_kl,
            bi_end_is_peak=self.bi_end_is_peak,
            bi_allow_sub_peak=self.bi_allow_sub_peak
        )
