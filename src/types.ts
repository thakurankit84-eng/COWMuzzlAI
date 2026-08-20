export interface BoundingBox {
  ymin: number; // 0 to 1000 or 0 to 1
  xmin: number;
  ymax: number;
  xmax: number;
}

export interface MuzzleDetection {
  id: string;
  box: BoundingBox; // normalized 0 to 1
  originalBox?: BoundingBox;
  appliedExpansionPercent?: number;
  confidence: number;
  label: string;
  classId: number;
  cropUrl?: string;
  biometricId: string;
  beadDensityScore: number; // 0-100
  ridgePatternClarity: 'High' | 'Medium' | 'Low';
  symmetryScore: number; // 0-100
  nostrilDistanceNorm: number;
}

export interface BreedCandidate {
  breed: string;
  scientificName?: string;
  speciesType: 'Bos indicus (Zebu)' | 'Bos taurus (Taurine)' | 'Crossbred' | 'Bos grunniens' | 'Other';
  confidence: number;
  purpose: 'Dairy' | 'Beef' | 'Dual-Purpose' | 'Draft / Working';
  origin: string;
  keyFeatures: string[];
  description: string;
}

export interface PhysicalTraits {
  coatColor: string;
  hornType: string;
  humpSize: 'Prominent' | 'Moderate' | 'Absent' | 'Small';
  dewlapSize: 'Large / Pendulous' | 'Moderate' | 'Tight / Minimal';
  earStructure: string;
  statureAndBuild: string;
  facialProfile: string;
}

export interface ProductionAndHealth {
  estimatedMilkYieldPerLactation?: string;
  milkFatPercentage?: string;
  climateTolerance: string;
  tickDiseaseResistance: 'High' | 'Moderate' | 'Low';
  temperament: string;
  recommendedCare: string[];
}

export interface MuzzleSpectralCrops {
  rgb: string;       // high-res RGB muzzle crop
  ridges: string;    // dermatoglyphic bead & ridge enhancement
  contrast: string;  // high contrast / CLAHE grayscale
  sobel: string;     // Sobel edge filter map
}

export interface RegistryValidationMatch {
  isMatch: boolean;
  matchedCowId?: string;
  matchedCowName?: string;
  similarityScore: number; // 0-100
  matchedTimestamp?: number;
  matchedScan?: CowAnalysisResult;
  matchReasons: string[];
}

export interface BiometricVectorEmbedding {
  vector: number[]; // 128-dimensional L2-normalized float embedding
  dimension: number; // 128
  norm: number; // 1.0 (L2 unit sphere)
  metric: 'cosine' | 'inner_product' | 'l2';
  subBands: {
    ridgeFrequencies: number[]; // dims 0..31
    beadDensityMap: number[]; // dims 32..63
    nasolabialSymmetry: number[]; // dims 64..95
    deepDermatoglyphics: number[]; // dims 96..127
  };
  extractedAt: number;
}

export interface FaissVectorRecord {
  id: string;
  cowId: string;
  breed: string;
  species: string;
  imageUrl: string;
  vector: number[];
  clusterId?: number;
  beadDensityScore?: number;
  symmetryScore?: number;
  timestamp: number;
  notes?: string;
}

export interface FaissSearchResult {
  rank: number;
  cowId: string;
  breed: string;
  species?: string;
  imageUrl?: string;
  similarityScore: number; // 0 to 100 percentage
  cosineSimilarity: number; // -1.0 to 1.0 (exact dot product for normalized vectors)
  l2Distance: number;
  matchConfidence: 'Definite Biometric Match' | 'Probable Match' | 'Breed / Strain Phenotype' | 'Distinct Animal';
  vectorRecord: FaissVectorRecord;
  subBandSimilarities: {
    ridgeScore: number; // 0 to 100
    beadScore: number;
    symmetryScore: number;
    deepTextureScore: number;
  };
  isCurrentQuery?: boolean;
}

export interface FaissIndexStats {
  totalVectors: number;
  dimension: number;
  indexType: 'IndexFlatIP' | 'IndexIVFFlat' | 'IndexFlatL2' | string;
  metricType: 'Cosine (Inner Product)' | 'L2 (Euclidean)' | string;
  nlist: number; // Voronoi cluster count
  nprobe: number;
  searchTimeMs: number;
  lastUpdated: number;
}

export interface ResNet50ModelInfo {
  name: string;
  filename: string;
  sizeBytes: number;
  sizeFormatted: string;
  googleDriveUrl: string;
  driveFileId: string;
  architecture: string;
  backbone: string;
  layersCount: number;
  inputShape: string;
  outputDimension: number; // 128-D / 2048-D
  embeddingType: string;
  task: 'feature_extraction' | 'biometric_embedding';
  status: 'Loaded & Active' | 'Ready' | 'Standby';
  description: string;
}

export interface AIModelsRegistryInfo {
  yolo: YOLOv8ModelInfo;
  resnet50: ResNet50ModelInfo;
}

export interface CowAnalysisResult {
  id: string;
  timestamp: number;
  imageUrl: string;
  annotatedImageUrl?: string;
  spectralCrops?: MuzzleSpectralCrops;
  imageDimensions?: { width: number; height: number };
  primaryBreed: BreedCandidate;
  alternateBreeds: BreedCandidate[];
  muzzleDetections: MuzzleDetection[];
  physicalTraits: PhysicalTraits;
  productionAndHealth: ProductionAndHealth;
  registryValidation?: RegistryValidationMatch;
  featureVector?: BiometricVectorEmbedding;
  faissSearchResults?: FaissSearchResult[];
  appliedBoxExpansion?: number;
  biometricPassport: {
    uniqueCattleId: string;
    muzzlePatternHash: string;
    registrationDate: string;
    verificationStatus: 'Verified Biometric' | 'Review Recommended';
  };
  modelMetadata: {
    yoloModelName: string;
    yoloWeightsFile: string;
    yoloDriveLink: string;
    resnet50ModelName: string;
    resnet50WeightsFile: string;
    resnet50DriveLink: string;
    inferenceTimeMs: number;
    modelVersion: string;
  };
}

export interface BreedDatabaseEntry {
  id: string;
  name: string;
  species: 'Bos indicus (Zebu)' | 'Bos taurus (Taurine)' | 'Crossbred' | string;
  category: 'Dairy' | 'Beef' | 'Dual-Purpose' | 'Draft' | string;
  origin: string;
  milkYield: string;
  fatContent: string;
  climate: string;
  description: string;
  characteristics: string[];
  imagePlaceholder: string;
  muzzleCharacteristics: string;
}

export interface YOLOv8ModelInfo {
  name: string;
  filename: string;
  sizeBytes: number;
  sizeFormatted: string;
  googleDriveUrl: string;
  driveFileId: string;
  architecture: string;
  baseModel: string;
  inputShape: string;
  classes: string[];
  task: string;
  status: 'Loaded & Active' | 'Ready' | 'Standby';
  description: string;
}
