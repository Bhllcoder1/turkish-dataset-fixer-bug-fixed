# turkish-dataset-fixer (Bug Fixed)

[serda-dev/turkish-dataset-fixer](https://github.com/serda-dev/turkish-dataset-fixer) projesindeki filtreleme kurallarını [turkish-nlp-suite/BellaTurca](https://huggingface.co/datasets/turkish-nlp-suite/BellaTurca) veri seti üzerinde test ettim. serda-dev'in pipeline'ındaki text normalization aşamasında bir bug buldum ve düzelttim.

## Olay Özeti

serda-dev LinguAI'den arkadaşım, birlikte çalışıyoruz. Kendisinin [turkish-dataset-fixer](https://github.com/serda-dev/turkish-dataset-fixer) projesi, Türkçe veri setlerini temizlemek için modüler bir pipeline sunuyor (normalizasyon, heuristik filtreler, dil doğrulama, KenLM, dedup). Ben bu pipeline'ın filtre kurallarını alıp [BellaTurca](https://huggingface.co/datasets/turkish-nlp-suite/BellaTurca) veri seti üzerinde denedim.

Pipeline'ı 2000 satırlık bir örneklemle çalıştırdığımda **hiçbir satır filtreyi geçemedi — %0**. BellaTurca kaliteli, editörlü bir veri seti olduğu için bu sonuç mantıklı değildi.

## Hatayı Nasıl Fark Ettim

Rapordaki istatistiklere baktığımda `too_few_words` filtresi 2000/2000 satırı yakalıyordu. Yani pipeline'a göre her metnin 5'ten az kelimesi vardı — bu imkansız, çünkü BellaTurca'daki metinler uzun makaleler.

Normalize edilmiş bir metni yazdırdığımda sorun ortaya çıktı:

```
Girdi:  "Bu bir test cümlesi"
Çıktı:  "Bubirbirtestcümlesi"   ← tüm boşluklar silinmiş
```

## serda-dev'in Kodundaki Hata

serda-dev'in `pipeline/text_normalization.py` dosyasındaki kontrol karakteri regex'i, normal boşluk karakterlerini (U+0020) de yakalayıp siliyordu:

```python
# serda-dev'in HATALI regex'i
_ctrl_re = re.compile(r"[^\S\r\n\t\u00A0-\uFFFF]|[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
```

`[^\S...]` deseni boşluk karakterini de hedefliyordu. Kelimeler birleşince metin tek dev kelimeye dönüşüyor, ardından:
- `too_few_words` → tek kelime var, 5'ten az → **reddet** (2000/2000)
- `avg_word_too_long` → ortalama kelime uzunluğu 45+ → **reddet** (1992/2000)
- `high_repeated_word` → en sık kelime oranı %100 → **reddet** (2000/2000)

Filtreler doğru çalışıyordu ama normalizasyon bozuk veri ürettiği için her şeyi reddediyordu.

## Düzeltmem

```python
# DÜZELTİLMİŞ — sadece gerçek kontrol karakterlerini siliyor
_ctrl_re = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
```

## Sonuç

| | serda-dev (hatalı) | Düzeltme Sonrası |
|---|---|---|
| Geçen | 0 (%0) | 1960 (%98) |
| Reddedilen | 2000 (%100) | 40 (%2) |

Düzeltme sonrası reddedilen 40 satırın dağılımı:
- 6 tanesi çok kısa (< 50 karakter)
- 24 tanesi Türkçe değil (dil doğrulama)
- 10 tanesi tam kopya (duplicate)

## Test Veri Seti

[turkish-nlp-suite/BellaTurca](https://huggingface.co/datasets/turkish-nlp-suite/BellaTurca) — **OzenliDerlem** subset, 2000 satır

## Filtre Aşamaları (serda-dev'den alınan)

| # | Aşama | Açıklama |
|---|-------|----------|
| 1 | Normalizasyon | Kontrol karakteri temizliği, Türkçe karakter koruması |
| 2 | Heuristik | Metin uzunluğu, alfabe oranı, tekrar, URL/HTML kontrolü |
| 3 | Dil Doğrulama | fastText + langdetect ile Türkçe kontrolü |
| 4 | Dedup | xxhash ile tam kopya tespiti |
| 5 | Kalite Skoru | Dil güveni + heuristik bileşenlerden soft skor |

> KenLM skoru atlandı — derlenmiş binary + eğitilmiş 5-gram model gerektirdiği için dahil edilmedi.

## Kullanım

```bash
pip install -r requirements.txt

# Varsayılan: 2000 satır
python filter_pipeline.py

# Özel satır sayısı ve çıktı dizini
python filter_pipeline.py --rows 5000 --output sonuclar
```

Çıktı `output/bellaturca_filtered.jsonl` dosyasına yazılır:

```json
{"text": "...", "lang": "tr", "lang_conf": 0.9998, "quality_score": 0.9754}
```

## Kullanılan Repolar ve Veri Setleri

| Kaynak | Link | Açıklama |
|--------|------|----------|
| Orijinal proje | [serda-dev/turkish-dataset-fixer](https://github.com/serda-dev/turkish-dataset-fixer) | Filtre kurallarının alındığı repo (bug burada) |
| Test veri seti | [turkish-nlp-suite/BellaTurca](https://huggingface.co/datasets/turkish-nlp-suite/BellaTurca) | OzenliDerlem subset, 2000 satır |
