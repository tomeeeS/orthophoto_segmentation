## Bemenetek
A deep-unet-for-satellite-image-segmentation-ben levő kódban a tanításhoz szükséges képek [innen](https://ieee-dataport.org/open-access/geonrw) tölthetőek le, 'data' nevű mappába kell rakni őket úgy, hogy azon belül 'images' nevűbe az eredeti képeket, 'elevation' nevűbe a magassági adatokat, és 'masks' nevűbe a referencia adatokat. A méretük miatt nem akarom velük terhelni a repot.  

## Kód
Eredetileg [ezzel](https://github.com/reachsumit/deep-unet-for-satellite-image-segmentation) a kódbázissal kezdtem dolgozni, ami műholdkép szegmentációt végzett; azóta ezt nagymértékben átdolgoztam és bővítettem.

### Forrásfájlok
Futtathatók: 
- **train_unet.py**: Neurális hálózatok tanítása
- **evaluate**: Betanított modellek kiértékelése a teszt képek halmazán (néhány száz kép, ill. azokból generált kisebb képrészlet (patch)), confusion matrix kirajzolása, lementése
- **predict**: Pár tesztképre való kirajzolás az eredeti, a referencia és a modell(ek) által adott eredménnyel
- **tpu-geonrw.ipynb**: Kaggle-ön, TPU-n való tanításhoz, egybe van benne ömlesztve pár fájlomból a tartalom, és módosítva hogy mehessen TPU-n. A Kaggle-re feltöltött adathalmazom amit használ [itt](https://www.kaggle.com/tomisajti/geonrw-patches) van.


Nem futtathatók:
- **cfg**: Általánosabb konstansok, állítható hiperparaméterek
- **losses**: Tanítás során használható hibafüggvények
- **unet_model** (nem saját kód): U-Net Keras modelles definíciója
- **patching**: Patch generálás az inputokból ill. referencia adatokból
- **data_handling**: Adat beolvasás, .tfrec fájl írás, data pipeline augmentációval
- **smooth_tiled_predictions** (nem saját kód): Eredmény kirajzolásahoz a patch-ekre való kimeneteket csúszóablakosan, egyszerre a patch-méretnél kevesebbet lépve rakjuk össze, nem csak a patchekre való kimeneteket egymás mellé rakva, mert olyankor a szélső pixeleknek nem lenne elég kontextusuk hogy elég jól be lehessen sorolni őket és négyzetrácsos hibák lennének az eredményben
- **to_objects**: Cseri Ádám kódja a neuronhálók minden pixelt kategóriához soroló kimenetéből "objektumokká" való konvertáláshoz
