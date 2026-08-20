import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { ImageUploadZone } from './components/ImageUploadZone';
import { MuzzleCanvasViewer } from './components/MuzzleCanvasViewer';
import { BiometricPassportCard } from './components/BiometricPassportCard';
import { RegistryValidationBanner } from './components/RegistryValidationBanner';
import { BiometricComparisonModal } from './components/BiometricComparisonModal';
import { YoloModelModal } from './components/YoloModelModal';
import { BreedEncyclopediaModal } from './components/BreedEncyclopediaModal';
import { HistoryRegistryDrawer } from './components/HistoryRegistryDrawer';
import { FaissVectorSearchModal } from './components/FaissVectorSearchModal';
import { BoundingBox, CowAnalysisResult, FaissVectorRecord, YOLOv8ModelInfo } from './types';
import { generateMultiSpectralImages, recomputeBiometricsWithExpandedBox, validateAgainstHerdRegistry } from './utils/muzzleBiometrics';
import { extractBiometricFeatureVector, globalFaissIndex } from './utils/faissEngine';
import { AlertCircle, Sparkles, RefreshCw, Layers, ShieldCheck, Cpu } from 'lucide-react';

export default function App() {
  const [modelInfo, setModelInfo] = useState<YOLOv8ModelInfo | null>(null);
  const [activeImageUrl, setActiveImageUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<CowAnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [isReTesting, setIsReTesting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Modals & Drawers
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const [isEncyclopediaModalOpen, setIsEncyclopediaModalOpen] = useState(false);
  const [isHistoryDrawerOpen, setIsHistoryDrawerOpen] = useState(false);
  const [isComparisonModalOpen, setIsComparisonModalOpen] = useState(false);
  const [isFaissModalOpen, setIsFaissModalOpen] = useState(false);

  // Saved herd scans (persisted in localStorage)
  const [savedScans, setSavedScans] = useState<CowAnalysisResult[]>(() => {
    try {
      const stored = localStorage.getItem('bovine_vision_saved_scans');
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  // Save to localStorage when updated
  useEffect(() => {
    try {
      localStorage.setItem('bovine_vision_saved_scans', JSON.stringify(savedScans));
    } catch (e) {
      console.error('Error saving scans to localStorage', e);
    }
  }, [savedScans]);

  // Load Model Information on mount and synchronize FAISS vector index with active registered scans
  useEffect(() => {
    fetch('/api/model-info')
      .then((res) => res.json())
      .then((data) => setModelInfo(data))
      .catch((err) => console.error('Failed to fetch model info:', err));

    // Clear server FAISS dummy presets and sync existing saved scans
    const syncSavedScansToFaiss = async () => {
      try {
        if (savedScans.length === 0) {
          // Ensure FAISS server starts clean
          await fetch('/api/faiss/index/clear', { method: 'POST' });
          globalFaissIndex.clear();
        } else {
          // Register saved cattle vectors into FAISS
          for (const scan of savedScans) {
            const vectorData = scan.featureVector?.vector || extractBiometricFeatureVector({
              seedHash: scan.biometricPassport.muzzlePatternHash,
              breed: scan.primaryBreed.breed,
              species: scan.primaryBreed.speciesType,
              beadDensityScore: scan.muzzleDetections?.[0]?.beadDensityScore || 88,
              symmetryScore: scan.muzzleDetections?.[0]?.symmetryScore || 90,
            }).vector;

            const record: FaissVectorRecord = {
              id: `vec-${scan.biometricPassport.uniqueCattleId}`,
              cowId: scan.biometricPassport.uniqueCattleId,
              breed: scan.primaryBreed.breed,
              species: scan.primaryBreed.speciesType,
              imageUrl: scan.imageUrl,
              vector: vectorData,
              beadDensityScore: scan.muzzleDetections?.[0]?.beadDensityScore || 88,
              symmetryScore: scan.muzzleDetections?.[0]?.symmetryScore || 90,
              timestamp: scan.timestamp,
            };

            globalFaissIndex.addVector(record);
            await fetch('/api/faiss/index/add', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ record }),
            });
          }
        }
      } catch (err) {
        console.warn('FAISS initial sync note:', err);
      }
    };

    syncSavedScansToFaiss();
  }, []);

  const handleAnalyzeImage = async (imageDataUrl: string, _sourceName?: string) => {
    setActiveImageUrl(imageDataUrl);
    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      // If URL, convert to base64 first if necessary, or pass directly
      let imagePayload = imageDataUrl;
      let mimeType = 'image/jpeg';

      if (imageDataUrl.startsWith('http')) {
        // Fetch image as blob and encode to base64
        try {
          const resp = await fetch(imageDataUrl);
          const blob = await resp.blob();
          mimeType = blob.type || 'image/jpeg';
          imagePayload = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result as string);
            reader.readAsDataURL(blob);
          });
        } catch {
          // If CORS prevents fetch, pass image URL directly
          imagePayload = imageDataUrl;
        }
      }

      const res = await fetch('/api/analyze-cow', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imagePayload, mimeType }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Analysis failed with status ${res.status}`);
      }

      const data: CowAnalysisResult = await res.json();
      const rawImage = data.imageUrl || imageDataUrl;

      // Generate all Multi-Spectral Images (Annotated, RGB Crop, Ridges, Contrast, Sobel)
      let annotatedImageUrl = rawImage;
      let spectralCrops = undefined;

      if (data.muzzleDetections && data.muzzleDetections.length > 0) {
        try {
          const multi = await generateMultiSpectralImages(rawImage, data.muzzleDetections[0]);
          annotatedImageUrl = multi.annotatedImageUrl;
          spectralCrops = multi.spectralCrops;
        } catch (cropErr) {
          console.warn('Could not generate offscreen multi-spectral crops:', cropErr);
        }
      }

      // Check for Biometric Fingerprint Match in existing Herd Registry
      const enrichedResult: CowAnalysisResult = {
        ...data,
        imageUrl: rawImage,
        annotatedImageUrl,
        spectralCrops,
        appliedBoxExpansion: data.appliedBoxExpansion ?? 0,
      };

      const validationMatch = validateAgainstHerdRegistry(enrichedResult, savedScans);
      enrichedResult.registryValidation = validationMatch;

      // If matched with an existing registered bovine, adopt the canonical cattle ID & biometric details
      if (validationMatch.isMatch && validationMatch.matchedCowId) {
        enrichedResult.biometricPassport.uniqueCattleId = validationMatch.matchedCowId;
        if (validationMatch.matchedScan?.biometricPassport?.muzzlePatternHash) {
          enrichedResult.biometricPassport.muzzlePatternHash = validationMatch.matchedScan.biometricPassport.muzzlePatternHash;
        }
      }

      setAnalysisResult(enrichedResult);
      setActiveImageUrl(rawImage);
    } catch (err: any) {
      console.error('Analysis error:', err);
      setErrorMessage(err.message || 'Failed to analyze cattle image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReTestWithBox = async (expansionPercent: number, effectiveBox: BoundingBox) => {
    if (!analysisResult) return;
    setIsReTesting(true);
    try {
      const updated = await recomputeBiometricsWithExpandedBox(
        analysisResult,
        expansionPercent,
        savedScans,
        effectiveBox
      );
      setAnalysisResult({ ...updated });

      // If this bovine was previously saved in the registry, synchronize its stored record
      setSavedScans((prevScans) => {
        const hasMatch = prevScans.some(
          (s) => s.id === updated.id || s.biometricPassport.uniqueCattleId === updated.biometricPassport.uniqueCattleId
        );
        if (!hasMatch) return prevScans;
        const next = prevScans.map((s) =>
          s.id === updated.id || s.biometricPassport.uniqueCattleId === updated.biometricPassport.uniqueCattleId
            ? { ...updated }
            : s
        );
        try {
          localStorage.setItem('bovine_vision_saved_scans', JSON.stringify(next));
        } catch {}
        return next;
      });
    } catch (err: any) {
      console.error('Re-test error:', err);
      setErrorMessage('Failed to re-test biometrics with expanded box: ' + (err.message || ''));
    } finally {
      setIsReTesting(false);
    }
  };

  const handleSaveToRegistry = async () => {
    if (!analysisResult) return;
    const targetCattleId =
      (analysisResult.registryValidation?.isMatch && analysisResult.registryValidation?.matchedCowId) ||
      analysisResult.biometricPassport.uniqueCattleId;

    const exists = savedScans.some(
      (s) =>
        s.id === analysisResult.id ||
        s.biometricPassport.uniqueCattleId === targetCattleId ||
        s.biometricPassport.uniqueCattleId === analysisResult.biometricPassport.uniqueCattleId
    );

    // Update local state and saved scans
    const newScans = exists
      ? savedScans.map((s) =>
          s.id === analysisResult.id ||
          s.biometricPassport.uniqueCattleId === targetCattleId ||
          s.biometricPassport.uniqueCattleId === analysisResult.biometricPassport.uniqueCattleId
            ? analysisResult
            : s
        )
      : [analysisResult, ...savedScans];

    setSavedScans(newScans);

    // Update validation on current result so banner immediately updates to registered status
    const updatedValidation = validateAgainstHerdRegistry(analysisResult, newScans);
    setAnalysisResult({
      ...analysisResult,
      registryValidation: updatedValidation,
    });

    // Persist 128-D vector to FAISS Server store & client index
    try {
      const vectorData = analysisResult.featureVector?.vector || extractBiometricFeatureVector({
        seedHash: analysisResult.biometricPassport.muzzlePatternHash,
        breed: analysisResult.primaryBreed.breed,
        species: analysisResult.primaryBreed.speciesType,
        beadDensityScore: analysisResult.muzzleDetections[0]?.beadDensityScore || 88,
        symmetryScore: analysisResult.muzzleDetections[0]?.symmetryScore || 90,
      }).vector;

      const record: FaissVectorRecord = {
        id: `vec-${analysisResult.biometricPassport.uniqueCattleId}`,
        cowId: analysisResult.biometricPassport.uniqueCattleId,
        breed: analysisResult.primaryBreed.breed,
        species: analysisResult.primaryBreed.speciesType,
        imageUrl: analysisResult.imageUrl,
        vector: vectorData,
        beadDensityScore: analysisResult.muzzleDetections[0]?.beadDensityScore || 88,
        symmetryScore: analysisResult.muzzleDetections[0]?.symmetryScore || 90,
        timestamp: Date.now(),
        notes: `Registered bovine profile on ${new Date().toLocaleDateString()}`,
      };

      globalFaissIndex.addVector(record);

      await fetch('/api/faiss/index/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ record }),
      });
    } catch (err) {
      console.warn('FAISS sync note:', err);
    }
  };

  const handleDeleteScan = async (id: string) => {
    const targetScan = savedScans.find((s) => s.id === id);
    const cowId = targetScan?.biometricPassport.uniqueCattleId;
    const remaining = savedScans.filter((s) => s.id !== id);
    setSavedScans(remaining);

    if (cowId) {
      globalFaissIndex.removeVector(`vec-${cowId}`);
      try {
        await fetch(`/api/faiss/index/vectors/${encodeURIComponent(cowId)}`, { method: 'DELETE' });
      } catch (err) {
        console.warn('FAISS vector deletion error:', err);
      }
    }

    if (analysisResult) {
      const updatedValidation = validateAgainstHerdRegistry(analysisResult, remaining);
      setAnalysisResult({
        ...analysisResult,
        registryValidation: updatedValidation,
      });
    }
  };

  const handleClearAllScans = async () => {
    if (window.confirm('Are you sure you want to clear all saved cattle scans and FAISS vector database records?')) {
      setSavedScans([]);
      localStorage.removeItem('bovine_vision_saved_scans');
      globalFaissIndex.clear();
      try {
        await fetch('/api/faiss/index/clear', { method: 'POST' });
      } catch (err) {
        console.warn('FAISS vector index clear error:', err);
      }

      if (analysisResult) {
        setAnalysisResult({
          ...analysisResult,
          registryValidation: validateAgainstHerdRegistry(analysisResult, []),
        });
      }
    }
  };

  const isCurrentScanSaved = Boolean(
    analysisResult && (
      savedScans.some(
        (s) =>
          s.id === analysisResult.id ||
          s.biometricPassport?.uniqueCattleId === analysisResult.biometricPassport?.uniqueCattleId
      ) ||
      (analysisResult.registryValidation?.isMatch &&
        analysisResult.registryValidation?.matchedCowId &&
        savedScans.some(
          (s) => s.biometricPassport?.uniqueCattleId === analysisResult.registryValidation?.matchedCowId
        ))
    )
  );

  return (
    <div className="min-h-screen bg-[#0a110a] text-slate-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-black relative overflow-x-hidden">
      {/* Ambient background glowing orbs for Frosted Glass theme */}
      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[55vw] h-[55vw] max-w-[700px] max-h-[700px] bg-emerald-900/25 rounded-full blur-[140px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[45vw] h-[45vw] max-w-[600px] max-h-[600px] bg-blue-900/20 rounded-full blur-[120px]"></div>
        <div className="absolute top-[35%] right-[15%] w-[35vw] h-[35vw] max-w-[500px] max-h-[500px] bg-lime-900/15 rounded-full blur-[100px]"></div>
        <div className="absolute top-[60%] left-[10%] w-[30vw] h-[30vw] max-w-[400px] max-h-[400px] bg-teal-900/15 rounded-full blur-[110px]"></div>
      </div>

      {/* Top Navigation Bar */}
      <Header
        modelInfo={modelInfo}
        onOpenModelModal={() => setIsModelModalOpen(true)}
        onOpenEncyclopediaModal={() => setIsEncyclopediaModalOpen(true)}
        onOpenHistoryDrawer={() => setIsHistoryDrawerOpen(true)}
        onOpenFaissModal={() => setIsFaissModalOpen(true)}
        savedCount={savedScans.length}
      />

      {/* Main Workspace Container */}
      <main className="relative z-10 flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6">
        {/* Error Alert if any */}
        {errorMessage && (
          <div className="backdrop-blur-xl bg-rose-950/40 border border-rose-500/30 text-rose-200 rounded-2xl p-4 flex items-start gap-3 shadow-lg">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <h4 className="text-sm font-bold text-rose-100">Analysis Warning</h4>
              <p className="text-xs mt-0.5 text-rose-300">{errorMessage}</p>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-rose-400 hover:text-rose-200 text-xs font-bold px-2 py-1 rounded-lg bg-rose-900/40 hover:bg-rose-900/60 transition-colors"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Section 1: Image Upload & Preset Selector Zone */}
        <section>
          <ImageUploadZone
            onImageSelected={handleAnalyzeImage}
            isAnalyzing={isAnalyzing}
            activeImageUrl={activeImageUrl}
          />
        </section>

        {/* Section 2: Registry Validation Banner (Match Status) */}
        {analysisResult && !isAnalyzing && (
          <RegistryValidationBanner
            analysis={analysisResult}
            onOpenComparison={() => setIsComparisonModalOpen(true)}
            onOpenFaissModal={() => setIsFaissModalOpen(true)}
            onSaveToRegistry={handleSaveToRegistry}
            isSaved={isCurrentScanSaved}
          />
        )}

        {/* Section 3: Interactive YOLOv8 Muzzle Detection Canvas & Biometric Crop */}
        {activeImageUrl && (
          <section className="space-y-3">
            <MuzzleCanvasViewer
              imageUrl={activeImageUrl}
              detections={analysisResult?.muzzleDetections || []}
              isAnalyzing={isAnalyzing}
              appliedBoxExpansion={analysisResult?.appliedBoxExpansion ?? 0}
              onReTestWithBox={handleReTestWithBox}
              isReTesting={isReTesting}
            />
          </section>
        )}

        {/* Section 4: Cattle Biometrics Identity Passport Card with Multi-Spectral Bundle */}
        {analysisResult && !isAnalyzing && (
          <div className="max-w-5xl mx-auto w-full">
            <BiometricPassportCard
              analysis={analysisResult}
              onSaveToRegistry={handleSaveToRegistry}
              onOpenFaissModal={() => setIsFaissModalOpen(true)}
              isSaved={isCurrentScanSaved}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="relative z-10 backdrop-blur-xl bg-white/[0.02] border-t border-white/10 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
            <span>AI-Based <span className="text-emerald-400 font-semibold">Cow Muzzle Identification</span> • YOLOv8 &amp; FAISS Vector Search</span>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsFaissModalOpen(true)}
              className="text-slate-400 hover:text-emerald-400 transition-colors font-mono cursor-pointer flex items-center gap-1"
            >
              <Cpu className="w-3 h-3 text-emerald-400" />
              <span>FAISS 128-D Cosine Search</span>
            </button>
            <span className="text-slate-600">•</span>
            <button
              onClick={() => setIsModelModalOpen(true)}
              className="text-slate-400 hover:text-emerald-400 transition-colors font-mono cursor-pointer"
            >
              YOLOv8 Weights
            </button>
            <span className="text-slate-600">•</span>
            <button
              onClick={() => setIsEncyclopediaModalOpen(true)}
              className="text-slate-400 hover:text-emerald-400 transition-colors cursor-pointer"
            >
              Breed Catalog
            </button>
          </div>
        </div>
      </footer>

      {/* Modals & Drawers */}
      <YoloModelModal
        isOpen={isModelModalOpen}
        onClose={() => setIsModelModalOpen(false)}
        modelInfo={modelInfo}
      />

      <BreedEncyclopediaModal
        isOpen={isEncyclopediaModalOpen}
        onClose={() => setIsEncyclopediaModalOpen(false)}
      />

      <FaissVectorSearchModal
        isOpen={isFaissModalOpen}
        onClose={() => setIsFaissModalOpen(false)}
        currentAnalysis={analysisResult}
        savedScans={savedScans}
        onSaveToRegistry={handleSaveToRegistry}
      />

      {analysisResult && analysisResult.registryValidation?.matchedScan && (
        <BiometricComparisonModal
          isOpen={isComparisonModalOpen}
          onClose={() => setIsComparisonModalOpen(false)}
          currentScan={analysisResult}
          matchedScan={analysisResult.registryValidation.matchedScan}
        />
      )}

      <HistoryRegistryDrawer
        isOpen={isHistoryDrawerOpen}
        onClose={() => setIsHistoryDrawerOpen(false)}
        savedScans={savedScans}
        onSelectScan={(scan) => {
          setActiveImageUrl(scan.imageUrl);
          setAnalysisResult(scan);
        }}
        onDeleteScan={handleDeleteScan}
        onClearAll={handleClearAllScans}
      />
    </div>
  );
}

