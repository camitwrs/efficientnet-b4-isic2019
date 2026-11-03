import random
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.comparacion import comparacion_pareada, wilcoxon_exacto


class TestWilcoxonExacto(unittest.TestCase):
    def test_cinco_positivas_da_el_minimo_declarado(self):
        # Con cinco diferencias no nulas, sin empates y del mismo signo, p = 2/2^5.
        w = wilcoxon_exacto([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertAlmostEqual(w["p_bilateral"], 0.0625)
        self.assertAlmostEqual(w["r_biserial"], 1.0)
        self.assertEqual(w["n_efectivo"], 5)

    def test_signo_negativo_favorece_a_m4(self):
        w = wilcoxon_exacto([-0.1, -0.2, -0.3, -0.4, -0.5])
        self.assertAlmostEqual(w["p_bilateral"], 0.0625)
        self.assertAlmostEqual(w["r_biserial"], -1.0)

    def test_caso_a_mano(self):
        # Rangos de |d| = 1,2,3,4,5; T+ = 1+3+4+5 = 13, S = 15, extremidad |13 - 7,5| = 5,5.
        # Patrones con |T+ - 7,5| >= 5,5: T+ en {0,1,2,13,14,15} -> 6 de 32.
        w = wilcoxon_exacto([0.1, -0.2, 0.3, 0.4, 0.5])
        self.assertEqual(w["t_mas"], 13)
        self.assertAlmostEqual(w["p_bilateral"], 6 / 32)
        self.assertAlmostEqual(w["r_biserial"], (13 - 2) / 15)

    def test_ceros_se_excluyen_tras_redondear(self):
        w = wilcoxon_exacto([0.0, 1e-14, 0.2, 0.3, 0.4])
        self.assertEqual(w["n_ceros"], 2)
        self.assertEqual(w["n_efectivo"], 3)
        self.assertAlmostEqual(w["p_bilateral"], 2 / 8)

    def test_empates_reciben_rango_medio(self):
        # |d| = 0,1 0,1 0,3 -> rangos 1,5 1,5 3; T+ = 1,5 + 3 = 4,5 (la segunda es negativa).
        w = wilcoxon_exacto([0.1, -0.1, 0.3])
        self.assertAlmostEqual(w["t_mas"], 4.5)
        self.assertAlmostEqual(w["suma_rangos"], 6.0)

    def test_todo_cero_no_aplica_contraste(self):
        w = wilcoxon_exacto([0.0, 0.0, 0.0])
        self.assertIsNone(w["p_bilateral"])
        self.assertIsNone(w["r_biserial"])

    def test_coincide_con_scipy_sin_empates(self):
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            self.skipTest("scipy no disponible")
        rng = random.Random(0)
        for _ in range(50):
            d = [rng.uniform(-1, 1) for _ in range(rng.randint(3, 9))]
            esperado = wilcoxon(d, alternative="two-sided", method="exact").pvalue
            self.assertAlmostEqual(wilcoxon_exacto(d)["p_bilateral"], esperado, places=10)


class TestComparacionPareada(unittest.TestCase):
    def test_diferencias_media_y_mediana(self):
        m1 = {42: 0.80, 123: 0.82, 456: 0.79}
        m4 = {42: 0.60, 123: 0.61, 456: 0.65}
        c = comparacion_pareada(m1, m4)
        self.assertAlmostEqual(c["pares"][42]["diferencia"], 0.20)
        self.assertAlmostEqual(c["media_diferencia"], (0.20 + 0.21 + 0.14) / 3)
        self.assertAlmostEqual(c["mediana_diferencia"], 0.20)

    def test_semillas_distintas_falla(self):
        with self.assertRaises(ValueError):
            comparacion_pareada({42: 1.0}, {123: 1.0})


if __name__ == "__main__":
    unittest.main()
