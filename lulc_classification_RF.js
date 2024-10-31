// LAST UPDATE - 31/10/2024
// Filter the collection for the VV product from the descending track
var collectionVV = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filterBounds(geometry)
    .select(['VV']);
print(collectionVV, 'Collection VV');

var collectionVH = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filterMetadata('resolution_meters','equals',10)
    .filterBounds(geometry)
    .select(['VH']);
// does not exist VH,HH

//filter by date
var SARVV=collectionVV.filterDate('2019-05-01','2019-05-31').mosaic();
var SARVH=collectionVH.filterDate('2019-05-01','2019-05-31').mosaic();
//Let's centre the map view over our ROI
Map.centerObject(geometry, 13);
//Filter to reduce speckle
var smothFactorRadius=50;
var SARVVFiltered=SARVV.focal_mean(smothFactorRadius,'circle','meters');
var SARVHFiltered=SARVH.focal_mean(smothFactorRadius,'circle','meters');


//FUNTIONS TO USE IN OPTICAL IMAGE
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}
//SENTINEL 2 CONSIDERING NOT CLOUD, DATE, AND GEOMETRY(LOCATION)
var datasetNoCloud= ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterDate('2019-05-01', '2019-05-20')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',30))
                  .filterBounds(geometry)
                  .map(maskS2clouds);
print(datasetNoCloud,'OPTICAFINAL')                  

var rescale = datasetNoCloud;
var image2 = rescale.median();
print(image2,'Imagen 2')
//Optical Image
var visualization ={
  bands:['B4','B3','B2'],// RGB Image
  min:0.0,
  max: 0.3,
};
Map.addLayer(image2.clip(geometry4), visualization, 'Sentinel_2');
Export.image.toDrive({             ///////EXPORT#2
  image: image2.select(['B2','B3','B4']).clip(geometry4),
  description:'Sentinel2_NoCloud_3B',
  folder:'GEE_2019',
  scale:10,
  region: geometry4});


var comp = image2;//(The original image has 13 bands, it is only the RGB)(Esta

var maskSAR = (comp.select('B4')).mask() //Máscara creada solo para exportar imagenes SAR
Map.addLayer(SARVVFiltered.mask(maskSAR).clip(geometry4),{min:-15,max:0},'SAR VV Filtered',0)
Map.addLayer(SARVHFiltered.mask(maskSAR).clip(geometry4),{min:-25,max:0},'SAR VH Filtered',0)

var comp = comp.clip(geometry4) //quitar esta linea del clip

//INDEX CALCULATION FOR SENTINEL_2 ///
//NDVI//   xq el ndvi si me descargo sin nubes el resto no
var ndvi = comp.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndviPalette = [
  '#FF0000', // Red
  '#FFA500', // Orange
  '#FFFF00', // Yellow
  '#9ACD32', // Yellow-Green
  '#32CD32'  // Green
]
Map.addLayer(ndvi,{palette:ndviPalette, min:-1, max:+1},'NDVI');
Export.image.toDrive({             ///////EXPORT#1
  image: ndvi,
  folder:'GEE_2019',
  description:'NDVI',
  scale:10,
  region: geometry4});
  
//NDBI//
var ndbi = comp.normalizedDifference(['B11', 'B8']).rename('NDBI');
Map.addLayer(ndbi, {min: -1, max: 1, palette: ['blue', 'white', 'red']}, 'NDBI');
Export.image.toDrive({             ///////EXPORT#2
  image: ndbi,
  description:'NDBI',
  folder:'GEE_2019',
  scale:10,
  region: geometry4});
//BU////
var bu = ndbi.subtract(ndvi);
Map.addLayer(bu, {min: -1, max: 1, palette: ['blue', 'white', 'red']}, 'BU1');
print(bu,'culito')
  
//NDWI// Normalized Difference Water Index
var ndwi = comp.normalizedDifference(['B3', 'B8']).rename('NDWI')
Map.addLayer(ndwi, {min: -1, max: 1, palette: ['blue', 'white', 'green']}, 'NDWI');
Export.image.toDrive({             ///////EXPORT#3
  image: ndwi,
  description:'NDWI',
  folder:'GEE_2019',
  scale:10,
  region: geometry4});
//SAVI// Soil Adjusted Vegetation Index s designed to minimize the soil background effect and improve vegetation discrimination
var L = 0.5
var savi = comp.expression(
    '((NIR - RED) * (1 + L)) / (NIR + RED + L)',
    {
      'NIR': comp.select('B8'),
      'RED': comp.select('B4'),
      'L': L
    }
  ).rename('SAVI');
Map.addLayer(savi, {min: -1, max: 1, palette: ['red', 'white', 'green']}, 'SAVI');
Export.image.toDrive({             ///////EXPORT#4
  image: savi,
  description:'SAVI',
  folder:'GEE_2019',
  scale:10,
  region: geometry4});
//EVI// Enhanced Vegetation Index
// Define a function to calculate EVI (Sentinel-2: B4 - Red, B8 - NIR, B2 - Blue)
  var L = 1;
  var C1 = 6;
  var C2 = 7.5;
  var G = 2.5;
  var evi = comp.expression(
    'G * ((NIR - RED) / (NIR + (C1 * RED) - (C2 * BLUE) + L))',
    {
      'NIR': comp.select('B8'),
      'RED': comp.select('B4'),
      'BLUE': comp.select('B2'),
      'L': L,
      'C1': C1,
      'C2': C2,
      'G': G
    }
     ).rename('EVI'); 
Map.addLayer(evi, {min: -1, max: 1, palette: ['red', 'white', 'green']}, 'EVI');
Export.image.toDrive({             ///////EXPORT#5
  image: evi,
  description:'EVI',
  folder:'GEE_2019',
  scale:10,
  region: geometry4});

//ARVI// // Atmospherically Resistant Vegetation Index 
//To calculate the Atmospherically Resistant Vegetation Index //is an improved index which is used to correct the influence of the atmosphere//
var arvi = comp.expression(
    '(NIR - (2 * RED) + BLUE) / (NIR + (2 * RED) + BLUE)',
    {
      'NIR': comp.select('B8'),
      'RED': comp.select('B4'),
      'BLUE': comp.select('B2')
    }
  ).rename('ARVI');
  Map.addLayer(arvi, {min: -1, max: 1, palette: ['red', 'white', 'green']}, 'ARVI');
  Export.image.toDrive({             ///////EXPORT#6
  image: arvi,
  description:'ARVI',
  folder:'GEE_2019',
  scale:10,
  region: geometry4});

//BSI//  write a code to calculation of "ARVI" index for Sentinel 2 in google earth engine javascript
//to capture soil variation
var bsi = comp.expression(
    '(RED + SWIR2) - (NIR + BLUE)',
    {
      'RED': comp.select('B4'),
      'NIR': comp.select('B8'),
      'BLUE': comp.select('B2'),
      'SWIR2': comp.select('B11')
    }
  ).rename('BSI');
Map.addLayer(bsi, {min: -1, max: 1, palette: ['blue', 'white', 'red']}, 'BSI');
Export.image.toDrive({             ///////EXPORT#7
  image: bsi,
  description:'BSI',
  folder:'GEE_2019',
  scale:10,
  region: geometry4});

//GCI//

//CI//

//INDEX CALCULATION FOR SENTINEL_1 ///
/// Radar Vegetation Index (RVI)
var rvi = SARVHFiltered.divide(SARVVFiltered.add(SARVHFiltered)).rename('RVI');
Map.addLayer (rvi, {min: 0, max: 1, palette: ['blue', 'white', 'green']}, 'RVI');
 
///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//var composite = ee.Image.cat(comp,ndvi); //the image actually has 14 bandas
var comp=image2
var composite= comp;
var bands2 = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'];
var input2 = composite.select(bands2); // nuestro input de images opticas será el input 2

//Join classes to be trained by RF algorithm
var training = WATER.merge(URBAN).merge(FOREST).merge(Shrubland).merge(GreenAreas);
var label = 'Class';
var landcoverPalette = [
  '#FF0000', //urban (1)
  '#ffeb6c', //Shrub and herb (2)
  '#056601', //forest (3)
  '#216aff', //water (4)
  '#98ff00', //green areas(5)
  ];
///////////////////////////////SAR TRAINING//////////////////////////////////////////////
// Define, train and run(ejecutar) the classification of SAR Images
var bands = ['VV','VH'];
var image = ee.Image.cat(SARVVFiltered,SARVHFiltered)
var input = image.select(bands);
var trainImage = input.sampleRegions({
  collection: training,
  properties: [label],
  scale: 30
});
var trainingData = trainImage.randomColumn();
var trainSet = trainingData.filter(ee.Filter.lessThan('random',0.8));//80 para entrenar, 20 validacion
var validationSet = trainingData.filter(ee.Filter.greaterThanOrEquals('random',0.8));
var mask = (input2.select('B4')).mask()
var maskxx=mask
var masked= input.mask(mask)
var classifier = ee.Classifier.smileRandomForest(10)
  .train({
    features: trainSet,
    classProperty: label,
    inputProperties: bands
  });
var classified = masked.classify(classifier);
Map.addLayer(classified.clip(geometry4),{palette: landcoverPalette, min:1, max:5},'Classification SAR');

Export.image.toDrive({             ///////EXPORT#7
  image: classified.clip(geometry4),
  description:'LC_SAR',
  folder:'GEE_2019',
  scale:10,
  region: geometry4});
// Classify the validation dataset using the trained classifier.
var validationClassified = validationSet.classify(classifier);
// Compute the accuracy of the classifier using a confusion matrix. ///donde indico el 20 y 80?
var confusionMatrix = validationClassified.errorMatrix(label, 'classification');
var overallAccuracy = confusionMatrix.accuracy();
//validation errot matrix and accurancy SAR
print('Error Matrix SAR',confusionMatrix)
print('Accuracy SAR',overallAccuracy)
print('Kappa Coeficiente SAR: ',confusionMatrix.kappa()); //Kappa Coeficient
var producersAccuracy = confusionMatrix.producersAccuracy();
print('Producers Accuracy:', producersAccuracy);
/////////////////////////////OPTICAL TRAINING///////////////////////////////////////////
// Define, train and run(ejecutar) the classification of Optical Image
var trainImage2=input2.sampleRegions({
  collection: training,
  properties: [label],
  scale: 30
});
var trainingData2 = trainImage2.randomColumn();
var trainSet2 = trainingData2.filter(ee.Filter.lessThan('random',0.8));
var validationSet2 = trainingData2.filter(ee.Filter.greaterThanOrEquals('random',0.8));
var classifier2 = ee.Classifier.smileRandomForest(10)
  .train({
    features: trainSet2,
    classProperty: label,
    inputProperties: bands2
  });
var classified2 = input2.classify(classifier2);
Map.addLayer(classified2.clip(geometry4),{palette: landcoverPalette, min:1, max:5},'classification Optical');



// Classify the validation dataset using the trained classifier.
var validationClassified = validationSet2.classify(classifier2);

// Compute the accuracy of the classifier using a confusion matrix.
var confusionMatrix = validationClassified.errorMatrix(label, 'classification');
var overallAccuracy = confusionMatrix.accuracy();
//validation errot matrix and accurancy OPTICO
print('Error Matrix Optico',confusionMatrix)
print('Accuracy Optico',overallAccuracy)
print('Kappa Coeficiente Optico: ',confusionMatrix.kappa()); //Kappa Coeficient
var producersAccuracy = confusionMatrix.producersAccuracy();
print('Producers Accuracy:', producersAccuracy);
/////////////////////////////COMBINATION OF SAR AND OPTIC TRAINING/////////////////////////////
// Creation of a combination of SAR image and Optical Image
var input3 = input2.addBands(masked.select(['VV','VH']))
print( input3,'input 3')

// Define, train and run(ejecutar) the classification of combined Image
var trainImage3 = input3.sampleRegions({
  collection: training,
  properties: [label],
  scale: 30
});
var trainingData3 = trainImage3.randomColumn();
var trainSet3 = trainingData3.filter(ee.Filter.lessThan('random',0.8));
var validationSet3 = trainingData3.filter(ee.Filter.greaterThanOrEquals('random',0.8));
var bands3=['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','VV','VH']
//Load the saved model from the Earth Engine asset
var classifier3 = ee.Classifier.smileRandomForest(100)
  .train({
    features: trainSet3,
    classProperty: label,
    inputProperties: bands3
  });
var classified3 = input3.classify(classifier3);
Map.addLayer(classified3,{palette: landcoverPalette, min:1, max:5},'classification combined');

///Efect to correct 
// Define the kernel for the majority filter.
var kernel = ee.Kernel.square({
  radius: 10,
  units: 'meters'
});

// Apply the majority filter to the classified image.
var majorityFiltered = classified3.focal_mode({
  kernel: kernel,
  iterations: 1
});
Map.addLayer(majorityFiltered.clip(geometry4),{palette: landcoverPalette, min:1, max:5},'classification combined without SaltPaper');


///////////////////////
////////////////////
/////////////////////////////////////////////
//////AIRE-SO2///////
var aoi= ee.Geometry(geometry4);
var collection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_SO2')
.select('SO2_column_number_density').filterDate('2019-04-01', '2019-06-30')
.filterBounds(aoi);
print (collection)
var so2 = collection.median().mask(mask).clip(aoi).rename('SO2');
/////////////////////////////////////////////
//////AIRE-O3///////
var collection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_O3')
.select('O3_column_number_density').filterDate('2019-04-01', '2019-06-30')
.filterBounds(aoi);
print (collection)
var o3 = collection.median().mask(mask).clip(aoi).rename('O3');
/////////////////AIRE---NO2//////////////////////////////////////
var collection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
.select('NO2_column_number_density').filterDate('2019-04-01', '2019-06-30')
.filterBounds(aoi);
print (collection)

var no2 = collection.median().mask(mask).clip(aoi).rename('NO2');


////////////Setlements
var cuenca= table.filter(ee.Filter.eq("DPA_DESPAR",'CUENCA'));
var geometriacuenca=cuenca.first()
var azogues= table.filter(ee.Filter.eq("DPA_DESPAR",'AZOGUES'));
var geometriaazogues=azogues.first()
var paute= table.filter(ee.Filter.eq("DPA_DESPAR",'PAUTE'));
var geometriapaute=paute.first()
var imgCuenca_2019 =majorityFiltered.clip(geometry4).clip(cuenca)
print()
var concatenada=ee.Image.cat(majorityFiltered.toDouble(),ndvi.toDouble(),ndbi.toDouble(),ndwi.toDouble(),savi.toDouble(),so2.toDouble(),o3.toDouble(),no2.toDouble())
Export.image.toDrive({
  image: concatenada.clip(geometry4).clip(cuenca),
  description:'Cuenca_2019comp',
  folder:'GEE_2019',
  scale:10,
  region: geometriacuenca});
  
var imgAzogues_2019 =majorityFiltered.clip(geometry4).clip(azogues)
Export.image.toDrive({
  image: concatenada.clip(geometry4).clip(azogues),
  description:'Azogues_2019comp',
  folder:'GEE_2019',
  scale:10,
  region: geometriaazogues});
  
var imgPaute_2019 =majorityFiltered.clip(geometry4).clip(paute)
Export.image.toDrive({
  image: concatenada.clip(geometry4).clip(paute),
  description:'Paute_2019comp',
  folder:'GEE_2019',
  scale:10,
  region: geometriapaute});

Map.addLayer(imgCuenca_2019,{palette: landcoverPalette, min:1, max:5},'Cuenca 2019');
Map.addLayer(imgAzogues_2019,{palette: landcoverPalette, min:1, max:5},'Azogues 2019');
Map.addLayer(imgPaute_2019,{palette: landcoverPalette, min:1, max:5},'Paute 2019');
//////////////////////////////////
/////////////////////////////////

Export.image.toDrive({
  image: majorityFiltered.clip(geometry4),
  description:'LC_2019',
  folder:'GEE_2019',
  scale:10,
  region: geometry4});

// Classify the validation dataset using the trained classifier.
var validationClassified = validationSet3.classify(classifier3);

// Compute the accuracy of the classifier using a confusion matrix.
var confusionMatrix = validationClassified.errorMatrix(label, 'classification');
var overallAccuracy = confusionMatrix.accuracy();

// Print the confusion matrix and overall accuracy to the console.
print('Confusion Matrix Validation Combined:', confusionMatrix);
print('Overall Accuracy Validation Combined:', overallAccuracy);
print('Kappa Coeficiente Combine: ',confusionMatrix.kappa()); //Kappa Coeficient
var consumersAccuracy = confusionMatrix.consumersAccuracy();
var producersAccuracy = confusionMatrix.producersAccuracy();
print('Consumers Accuracy:', consumersAccuracy);
print('Producers Accuracy:', producersAccuracy);
/////////////////////////////////////////////////////////////////////////////////////////////////


/*savemodel
var assetId = "users/guamanpamela/random_forest_model";
Export.table.toAsset({
  collection: ee.FeatureCollection(ee.Feature(geometry, { "classifier": classifier3.serialize() })),
  description: "random_forest_model",
  assetId: assetId
});
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////APPLICATION OF THE RF MODEL ////////////////////////////////////////////

// Filter the collection for the VV product from the descending track
var collectionVV = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filterBounds(geometry)
    .select(['VV']);
print(collectionVV, 'Collection VV');

var collectionVH = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filterMetadata('resolution_meters','equals',10)
    .filterBounds(geometry)
    .select(['VH']);
//filter by date
var SARVV=collectionVV.filterDate('2022-01-01','2022-01-31').mosaic();
var SARVH=collectionVH.filterDate('2022-01-01','2022-01-31').mosaic();
print(SARVV,'SAR_VV_2part')
print(SARVH,'SAR_VH_2part')
Export.image.toDrive({             ///////EXPORT#2
  image: ndbi,
  description:'SARVV',
  folder:'GEE_2022',
  scale:10,
  region: geometry4});
//Filter to reduce speckle
var smothFactorRadius=50;
var SARVVFiltered=SARVV.focal_mean(smothFactorRadius,'circle','meters');
var SARVHFiltered=SARVH.focal_mean(smothFactorRadius,'circle','meters');
Map.addLayer(SARVVFiltered,{min:-15,max:0},'SAR VV Filtered SecPart',0)
Map.addLayer(SARVHFiltered,{min:-25,max:0},'SAR VH Filtered SecPart',0)

//Sentinel 2 collection .filterDate()
var datasetNoCloud = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterDate('2022-01-10', '2022-01-12')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',40))
                  .filterBounds(geometry)
                  .map(maskS2clouds);
var rescale = datasetNoCloud;
var image2 = rescale.median();

//Gráfico de Imagen Optica
var visualization ={
  bands:['B4','B3','B2'],//RGB Image
  min:0.0,
  max: 0.3,
};
Map.addLayer(image2, visualization, 'Sentinel_2_SecondPart');

var comp = image2;//la imagen original tiene 13 bandas
var comp = comp.clip(geometry4)
var ndvi = comp.normalizedDifference(['B8', 'B4']).rename('NDVI'); ////como hago si quiero agregar mas indices?? osea concateno mas 

//INDEX CALCULATION FOR SENTINEL_2 -SECOND PART ///
//NDVI//   xq el ndvi si me descargo sin nubes el resto no
var ndvi = comp.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndviPalette = [
  '#FF0000', // Red
  '#FFA500', // Orange
  '#FFFF00', // Yellow
  '#9ACD32', // Yellow-Green
  '#32CD32'  // Green
]
Map.addLayer(ndvi,{palette:ndviPalette, min:-1, max:+1},'NDVI');
Export.image.toDrive({             ///////EXPORT#1
  image: ndvi,
  folder:'GEE_2022',
  description:'NDVI_2',
  scale:10,
  region: geometry4});
  
//NDBI//
var ndbi = comp.normalizedDifference(['B11', 'B8']).rename('NDBI');
Map.addLayer(ndbi, {min: -1, max: 1, palette: ['blue', 'white', 'red']}, 'NDBI');
Export.image.toDrive({             ///////EXPORT#2
  image: ndbi,
  description:'NDBI_2',
  folder:'GEE_2022',
  scale:10,
  region: geometry4});

 var bu = ndbi - ndvi
 Map.addLayer(bu, {min: -2, max: 2, palette: ['blue', 'white', 'red']}, 'BU');
//NDWI// Normalized Difference Water Index
var ndwi = comp.normalizedDifference(['B3', 'B8']).rename('NDWI')
Map.addLayer(ndwi, {min: -1, max: 1, palette: ['blue', 'white', 'green']}, 'NDWI');
Export.image.toDrive({             ///////EXPORT#3
  image: ndwi,
  description:'NDWI_2',
  folder:'GEE_2022',
  scale:10,
  region: geometry4});
//SAVI// Soil Adjusted Vegetation Index s designed to minimize the soil background effect and improve vegetation discrimination
var L = 0.5
var savi = comp.expression(
    '((NIR - RED) * (1 + L)) / (NIR + RED + L)',
    {
      'NIR': comp.select('B8'),
      'RED': comp.select('B4'),
      'L': L
    }
  ).rename('SAVI');
Map.addLayer(savi, {min: -1, max: 1, palette: ['red', 'white', 'green']}, 'SAVI');
Export.image.toDrive({             ///////EXPORT#4
  image: savi,
  description:'SAVI_2',
  folder:'GEE_2022',
  scale:10,
  region: geometry4});
//EVI// Enhanced Vegetation Index
// Define a function to calculate EVI (Sentinel-2: B4 - Red, B8 - NIR, B2 - Blue)
  var L = 1;
  var C1 = 6;
  var C2 = 7.5;
  var G = 2.5;
  var evi = comp.expression(
    'G * ((NIR - RED) / (NIR + (C1 * RED) - (C2 * BLUE) + L))',
    {
      'NIR': comp.select('B8'),
      'RED': comp.select('B4'),
      'BLUE': comp.select('B2'),
      'L': L,
      'C1': C1,
      'C2': C2,
      'G': G
    }
     ).rename('EVI'); 
Map.addLayer(evi, {min: -1, max: 1, palette: ['red', 'white', 'green']}, 'EVI');
Export.image.toDrive({             ///////EXPORT#5
  image: evi,
  description:'EVI_2',
  folder:'GEE_2022',
  scale:10,
  region: geometry4});

//ARVI// // Atmospherically Resistant Vegetation Index 
//To calculate the Atmospherically Resistant Vegetation Index //is an improved index which is used to correct the influence of the atmosphere//
var arvi = comp.expression(
    '(NIR - (2 * RED) + BLUE) / (NIR + (2 * RED) + BLUE)',
    {
      'NIR': comp.select('B8'),
      'RED': comp.select('B4'),
      'BLUE': comp.select('B2')
    }
  ).rename('ARVI');
  Map.addLayer(arvi, {min: -1, max: 1, palette: ['red', 'white', 'green']}, 'ARVI');
  Export.image.toDrive({             ///////EXPORT#6
  image: arvi,
  description:'ARVI_2',
  folder:'GEE_2022',
  scale:10,
  region: geometry4});

//BSI//  write a code to calculation of "ARVI" index for Sentinel 2 in google earth engine javascript
//to capture soil variation
var bsi = comp.expression(
    '(RED + SWIR2) - (NIR + BLUE)',
    {
      'RED': comp.select('B4'),
      'NIR': comp.select('B8'),
      'BLUE': comp.select('B2'),
      'SWIR2': comp.select('B11')
    }
  ).rename('BSI');
Map.addLayer(bsi, {min: -1, max: 1, palette: ['blue', 'white', 'red']}, 'BSI');
Export.image.toDrive({             ///////EXPORT#7
  image: bsi,
  description:'BSI_2',
  folder:'GEE_2022',
  scale:10,
  region: geometry4});

//GCI//

//CI//
var comp= image2
/////////////////////////////////////////////////////////////////////////////////
var composite = ee.Image.cat(comp,ndvi); //La iamgen ahora tiene 14 bandas
var bands2 = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'];
var input2 = comp.select(bands2); // nuestro input de images opticas será el input 2

//unimos todas las classes a ser entrenadas por el algorithmo
var label = 'Class';
var landcoverPalette = [
  '#FF0000', //urban (1)
  '#ffeb6c', //Shrub and herb (2)
  '#056601', //forest (3)
  '#216aff', //water (4)
  '#98ff00', //green areas(5)
  ];

// Aquí vamos a definir, entrenar y ejecutar la clasificacion de las imagenes SAR
var bands = ['VV','VH'];
var image = ee.Image.cat(SARVVFiltered,SARVHFiltered)
var input = image.select(bands);

var mask = (input2.select('B4')).mask()
var masked= input.mask(mask)

// Aquí se crea una combinación de imagenes ópticas y SAR
var input3 = input2.addBands(masked.select(['VV','VH']))
var classified3 = input3.classify(classifier3);
Map.addLayer(classified3,{palette: landcoverPalette, min:1, max:5},'classification combinada_SecondPart');
///Efect to correct 
// Define the kernel for the majority filter.
var kernel = ee.Kernel.square({
  radius: 10,
  units: 'meters'
});

// Apply the majority filter to the classified image.
var majorityFilteredSecondPart = classified3.focal_mode({
  kernel: kernel,
  iterations: 1
});