import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(RoadDamageApp(cameras: cameras));
}

class RoadDamageApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const RoadDamageApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Road Damage Detector',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFFE53935),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: HomeScreen(cameras: cameras),
    );
  }
}

// ─────────────────────────────────────────────
// Home Screen
// ─────────────────────────────────────────────

class HomeScreen extends StatelessWidget {
  final List<CameraDescription> cameras;
  const HomeScreen({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF121212),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.report_problem_rounded,
                  size: 80, color: Color(0xFFE53935)),
              const SizedBox(height: 16),
              const Text(
                'Road Damage\nDetector',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                  height: 1.2,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Detects: Cracks • Potholes • Surface Distress',
                style: TextStyle(color: Colors.grey[400], fontSize: 14),
              ),
              const SizedBox(height: 56),
              _ActionButton(
                icon: Icons.camera_alt_rounded,
                label: 'Use Camera',
                subtitle: 'Capture and detect',
                color: const Color(0xFFE53935),
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => CameraDetectionScreen(cameras: cameras),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              _ActionButton(
                icon: Icons.photo_library_rounded,
                label: 'Upload Image',
                subtitle: 'Pick from gallery',
                color: const Color(0xFF1E88E5),
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => const ImageDetectionScreen(),
                  ),
                ),
              ),
              const SizedBox(height: 40),
              _buildLegend(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLegend() {
    final classes = [
      ('Pothole', Colors.red),
      ('Crack', Colors.orange),
      ('Surface Distress', Colors.yellow),
    ];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Detection Classes',
            style: TextStyle(
                color: Colors.grey[400],
                fontSize: 12,
                fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        Wrap(
          alignment: WrapAlignment.center,
          spacing: 12,
          runSpacing: 6,
          children: classes
              .map((c) => Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                          width: 10,
                          height: 10,
                          decoration: BoxDecoration(
                              color: c.$2, shape: BoxShape.circle)),
                      const SizedBox(width: 4),
                      Text(c.$1,
                          style:
                              TextStyle(color: Colors.grey[300], fontSize: 12)),
                    ],
                  ))
              .toList(),
        ),
      ],
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final String subtitle;
  final Color color;
  final VoidCallback onTap;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.subtitle,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: color.withOpacity(0.15),
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                    color: color.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(12)),
                child: Icon(icon, color: color, size: 28),
              ),
              const SizedBox(width: 16),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(label,
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w600)),
                  Text(subtitle,
                      style: TextStyle(color: Colors.grey[400], fontSize: 13)),
                ],
              ),
              const Spacer(),
              Icon(Icons.arrow_forward_ios_rounded,
                  color: Colors.grey[600], size: 16),
            ],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────
// Detection model
// ─────────────────────────────────────────────

class Detection {
  final int classId;
  final double confidence;
  final Rect boundingBox;
  final String label;
  final Color color;

  const Detection({
    required this.classId,
    required this.confidence,
    required this.boundingBox,
    required this.label,
    required this.color,
  });
}

class RoadDamageDetector {
  static const List<String> classNames = [
    'Pothole',
    'Crack',
    'Surface Distress',
  ];
  static const List<Color> classColors = [
    Colors.red,
    Colors.orange,
    Colors.yellow,
  ];

  static const int inputSize = 640;
  static const double confThreshold = 0.35;
  static const double iouThreshold = 0.5;
  static const int numClasses = 3;
  static const int outputRows = 7; // 4 bbox + 3 classes

  Interpreter? _interpreter;
  bool _isLoaded = false;

  Future<void> loadModel() async {
    try {
      final byteData =
          await rootBundle.load('assets/road_damage_50epochs.tflite');
      final tempDir = await getTemporaryDirectory();
      final file = File('${tempDir.path}/road_damage_model.tflite');
      await file.writeAsBytes(byteData.buffer.asUint8List());
      _interpreter = Interpreter.fromFile(file);
      _isLoaded = true;
      debugPrint('✅ Model loaded successfully');
    } catch (e) {
      _isLoaded = false;
      debugPrint('❌ Failed to load model: $e');
    }
  }

  bool get isLoaded => _isLoaded;

  Future<List<Detection>> detect(File imageFile) async {
    if (!_isLoaded || _interpreter == null) {
      debugPrint('⚠️ Model not loaded, skipping detection');
      return [];
    }

    try {
      final bytes = await imageFile.readAsBytes();
      final original = img.decodeImage(bytes);
      if (original == null) return [];

      final resized =
          img.copyResize(original, width: inputSize, height: inputSize);

      final input = List.generate(
        1,
        (_) => List.generate(
          inputSize,
          (y) => List.generate(inputSize, (x) {
            final pixel = resized.getPixel(x, y);
            return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
          }),
        ),
      );

      // Output shape: [1, 7, 8400]
      final output = List.generate(
          1, (_) => List.generate(outputRows, (_) => List.filled(8400, 0.0)));

      _interpreter!.run(input, output);

      // Transpose [7, 8400] → [8400, 7]
      final transposed = List.generate(
          8400, (i) => List.generate(outputRows, (j) => output[0][j][i]));

      final detections = _parseOutput(transposed);
      return _nms(detections);
    } catch (e) {
      debugPrint('❌ Detection error: $e');
      return [];
    }
  }

  List<Detection> _parseOutput(List<List<double>> output) {
    final detections = <Detection>[];
    for (final pred in output) {
      final x1 = pred[0] / inputSize;
      final y1 = pred[1] / inputSize;
      final x2 = pred[2] / inputSize;
      final y2 = pred[3] / inputSize;

      double bestConf = 0;
      int bestClass = 0;
      for (int c = 0; c < numClasses; c++) {
        if (pred[4 + c] > bestConf) {
          bestConf = pred[4 + c];
          bestClass = c;
        }
      }
      if (bestConf < confThreshold) continue;

      detections.add(Detection(
        classId: bestClass,
        confidence: bestConf,
        boundingBox: Rect.fromLTRB(
          x1.clamp(0.0, 1.0),
          y1.clamp(0.0, 1.0),
          x2.clamp(0.0, 1.0),
          y2.clamp(0.0, 1.0),
        ),
        label: classNames[bestClass],
        color: classColors[bestClass],
      ));
    }
    return detections;
  }

  List<Detection> _nms(List<Detection> detections) {
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));
    final kept = <Detection>[];
    for (final det in detections) {
      bool suppressed = false;
      for (final keep in kept) {
        if (det.classId == keep.classId &&
            _iou(det.boundingBox, keep.boundingBox) > iouThreshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) kept.add(det);
    }
    return kept;
  }

  double _iou(Rect a, Rect b) {
    final il = a.left > b.left ? a.left : b.left;
    final it = a.top > b.top ? a.top : b.top;
    final ir = a.right < b.right ? a.right : b.right;
    final ib = a.bottom < b.bottom ? a.bottom : b.bottom;
    if (ir <= il || ib <= it) return 0;
    final inter = (ir - il) * (ib - it);
    final union = a.width * a.height + b.width * b.height - inter;
    return union == 0 ? 0 : inter / union;
  }

  void dispose() => _interpreter?.close();
}

// ─────────────────────────────────────────────
// Bounding box painter
// ─────────────────────────────────────────────

class BoundingBoxPainter extends CustomPainter {
  final List<Detection> detections;
  const BoundingBoxPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    for (final det in detections) {
      final rect = Rect.fromLTWH(
        det.boundingBox.left * size.width,
        det.boundingBox.top * size.height,
        det.boundingBox.width * size.width,
        det.boundingBox.height * size.height,
      );

      // Box stroke
      canvas.drawRect(
        rect,
        Paint()
          ..color = det.color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.5,
      );

      // Label
      final labelText =
          ' ${det.label} ${(det.confidence * 100).toStringAsFixed(0)}% ';
      final tp = TextPainter(
        text: TextSpan(
          text: labelText,
          style: const TextStyle(
              color: Colors.white, fontSize: 11, fontWeight: FontWeight.bold),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final labelTop = rect.top - tp.height - 2;
      canvas.drawRect(
        Rect.fromLTWH(rect.left, labelTop, tp.width, tp.height + 2),
        Paint()..color = det.color.withOpacity(0.85),
      );
      tp.paint(canvas, Offset(rect.left, labelTop + 1));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

// ─────────────────────────────────────────────
// Detection summary widget
// ─────────────────────────────────────────────

class _DetectionSummary extends StatelessWidget {
  final List<Detection> detections;
  const _DetectionSummary({required this.detections});

  @override
  Widget build(BuildContext context) {
    final counts = <String, int>{};
    final colors = <String, Color>{};
    final topConf = <String, double>{};

    for (final d in detections) {
      counts[d.label] = (counts[d.label] ?? 0) + 1;
      colors[d.label] = d.color;
      if ((topConf[d.label] ?? 0) < d.confidence) {
        topConf[d.label] = d.confidence;
      }
    }

    return Container(
      color: const Color(0xFF1E1E1E),
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Total header ──
          Row(
            children: [
              const Icon(Icons.warning_amber_rounded,
                  color: Color(0xFFE53935), size: 18),
              const SizedBox(width: 8),
              Text(
                '${detections.length} detection${detections.length == 1 ? '' : 's'} found',
                style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 15),
              ),
            ],
          ),
          const SizedBox(height: 12),

          // ── Per-class summary cards ──
          ...counts.entries.map((e) {
            final color = colors[e.key]!;
            final best = topConf[e.key]!;
            return Container(
              margin: const EdgeInsets.only(bottom: 8),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              decoration: BoxDecoration(
                color: color.withOpacity(0.08),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: color.withOpacity(0.3)),
              ),
              child: Row(
                children: [
                  Container(
                      width: 12,
                      height: 12,
                      decoration:
                          BoxDecoration(color: color, shape: BoxShape.circle)),
                  const SizedBox(width: 10),
                  Text(e.key,
                      style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                          fontSize: 14)),
                  const Spacer(),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                    decoration: BoxDecoration(
                      color: color.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: color.withOpacity(0.5)),
                    ),
                    child: Text('${e.value}x',
                        style: TextStyle(
                            color: color,
                            fontSize: 12,
                            fontWeight: FontWeight.bold)),
                  ),
                  const SizedBox(width: 10),
                  Text(
                    'Top: ${(best * 100).toStringAsFixed(0)}%',
                    style: TextStyle(color: Colors.grey[400], fontSize: 12),
                  ),
                ],
              ),
            );
          }),

          const Divider(color: Colors.white12, height: 16),

          // ── Top detections list ──
          Text('Top detections',
              style: TextStyle(
                  color: Colors.grey[500],
                  fontSize: 11,
                  fontWeight: FontWeight.bold)),
          const SizedBox(height: 6),
          ...detections.take(8).map((d) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 3),
                child: Row(
                  children: [
                    Container(
                        width: 8,
                        height: 8,
                        decoration: BoxDecoration(
                            color: d.color, shape: BoxShape.circle)),
                    const SizedBox(width: 8),
                    Text(d.label,
                        style:
                            TextStyle(color: Colors.grey[300], fontSize: 13)),
                    const Spacer(),
                    Text('${(d.confidence * 100).toStringAsFixed(1)}%',
                        style:
                            TextStyle(color: Colors.grey[500], fontSize: 12)),
                  ],
                ),
              )),
          if (detections.length > 8)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                '+ ${detections.length - 8} more',
                style: TextStyle(color: Colors.grey[600], fontSize: 11),
              ),
            ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────
// Image upload detection screen
// ─────────────────────────────────────────────

class ImageDetectionScreen extends StatefulWidget {
  const ImageDetectionScreen({super.key});

  @override
  State<ImageDetectionScreen> createState() => _ImageDetectionScreenState();
}

class _ImageDetectionScreenState extends State<ImageDetectionScreen> {
  final RoadDamageDetector _detector = RoadDamageDetector();
  File? _imageFile;
  List<Detection> _detections = [];
  bool _loading = false;
  bool _modelReady = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    await _detector.loadModel();
    if (mounted) setState(() => _modelReady = _detector.isLoaded);
  }

  @override
  void dispose() {
    _detector.dispose();
    super.dispose();
  }

  Future<void> _pickImage() async {
    if (!_modelReady) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Model is still loading, please wait...'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() {
      _imageFile = File(picked.path);
      _loading = true;
      _detections = [];
    });

    final detections = await _detector.detect(_imageFile!);
    if (mounted) {
      setState(() {
        _detections = detections;
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF121212),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E1E1E),
        foregroundColor: Colors.white,
        title: const Text('Image Detection'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: Icon(
              _modelReady ? Icons.check_circle : Icons.hourglass_bottom,
              color: _modelReady ? Colors.green : Colors.orange,
              size: 20,
            ),
          ),
          IconButton(
            icon: const Icon(Icons.photo_library_rounded),
            onPressed: _pickImage,
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // ── Image area ──
            SizedBox(
              height: MediaQuery.of(context).size.height * 0.45,
              child: _imageFile == null
                  ? Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.add_photo_alternate_rounded,
                              size: 64, color: Colors.grey[700]),
                          const SizedBox(height: 12),
                          Text(
                            _modelReady
                                ? 'Tap the icon above to pick an image'
                                : 'Loading model...',
                            style: TextStyle(color: Colors.grey[500]),
                          ),
                          if (!_modelReady) ...[
                            const SizedBox(height: 12),
                            const CircularProgressIndicator(
                                color: Color(0xFFE53935)),
                          ]
                        ],
                      ),
                    )
                  : _loading
                      ? const Center(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              CircularProgressIndicator(
                                  color: Color(0xFFE53935)),
                              SizedBox(height: 16),
                              Text('Analyzing...',
                                  style: TextStyle(color: Colors.white)),
                            ],
                          ),
                        )
                      : Stack(
                          fit: StackFit.expand,
                          children: [
                            Image.file(_imageFile!, fit: BoxFit.contain),
                            CustomPaint(
                                painter: BoundingBoxPainter(_detections)),
                          ],
                        ),
            ),

            // ── Results ──
            if (!_loading && _imageFile != null) ...[
              if (_detections.isNotEmpty)
                _DetectionSummary(detections: _detections)
              else
                Container(
                  color: const Color(0xFF1E1E1E),
                  padding: const EdgeInsets.all(16),
                  child: const Row(
                    children: [
                      Icon(Icons.check_circle_rounded, color: Colors.green),
                      SizedBox(width: 8),
                      Text('No road damage detected',
                          style: TextStyle(color: Colors.white)),
                    ],
                  ),
                ),
            ],
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────
// Camera detection screen
// ─────────────────────────────────────────────

class CameraDetectionScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const CameraDetectionScreen({super.key, required this.cameras});

  @override
  State<CameraDetectionScreen> createState() => _CameraDetectionScreenState();
}

class _CameraDetectionScreenState extends State<CameraDetectionScreen> {
  CameraController? _cameraController;
  final RoadDamageDetector _detector = RoadDamageDetector();
  bool _isProcessing = false;
  bool _cameraReady = false;
  bool _modelReady = false;
  List<Detection> _detections = [];
  File? _lastFrame;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _loadModel();
  }

  Future<void> _loadModel() async {
    await _detector.loadModel();
    if (mounted) setState(() => _modelReady = _detector.isLoaded);
  }

  Future<void> _initCamera() async {
    if (widget.cameras.isEmpty) return;
    try {
      _cameraController = CameraController(
        widget.cameras.first,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await _cameraController!.initialize();
      if (mounted) setState(() => _cameraReady = true);
    } catch (e) {
      debugPrint('❌ Camera init error: $e');
    }
  }

  Future<void> _captureAndDetect() async {
    if (_isProcessing) return;
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Camera not ready yet'),
            backgroundColor: Colors.orange),
      );
      return;
    }
    if (!_modelReady) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Model still loading, please wait...'),
            backgroundColor: Colors.orange),
      );
      return;
    }

    setState(() => _isProcessing = true);
    try {
      final xFile = await _cameraController!.takePicture();
      final imageFile = File(xFile.path);
      final detections = await _detector.detect(imageFile);
      if (mounted) {
        setState(() {
          _lastFrame = imageFile;
          _detections = detections;
        });
      }
    } catch (e) {
      debugPrint('❌ Capture error: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content: Text('Capture failed: $e'), backgroundColor: Colors.red),
        );
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _detector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: const Text('Camera Detection'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Row(
              children: [
                Icon(
                  _modelReady ? Icons.check_circle : Icons.hourglass_bottom,
                  color: _modelReady ? Colors.green : Colors.orange,
                  size: 18,
                ),
                const SizedBox(width: 4),
                Text(
                  _modelReady ? 'Ready' : 'Loading...',
                  style: TextStyle(
                      color: _modelReady ? Colors.green : Colors.orange,
                      fontSize: 12),
                ),
              ],
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          // ── Camera preview ──
          Expanded(
            child: !_cameraReady
                ? const Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(color: Color(0xFFE53935)),
                        SizedBox(height: 12),
                        Text('Initializing camera...',
                            style: TextStyle(color: Colors.white)),
                      ],
                    ),
                  )
                : Stack(
                    fit: StackFit.expand,
                    children: [
                      CameraPreview(_cameraController!),
                      if (_lastFrame != null)
                        CustomPaint(painter: BoundingBoxPainter(_detections)),
                    ],
                  ),
          ),

          // ── Summary after capture ──
          if (_detections.isNotEmpty)
            _DetectionSummary(detections: _detections),

          if (_lastFrame != null && _detections.isEmpty && !_isProcessing)
            Container(
              color: const Color(0xFF1A1A1A),
              padding: const EdgeInsets.all(12),
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.check_circle_rounded, color: Colors.green),
                  SizedBox(width: 8),
                  Text('No road damage detected',
                      style: TextStyle(color: Colors.white)),
                ],
              ),
            ),

          // ── Capture button ──
          Container(
            color: Colors.black,
            padding: const EdgeInsets.all(20),
            child: GestureDetector(
              onTap: _captureAndDetect,
              child: Container(
                width: 70,
                height: 70,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _isProcessing
                      ? Colors.grey
                      : !_modelReady
                          ? Colors.orange
                          : const Color(0xFFE53935),
                  border: Border.all(color: Colors.white, width: 3),
                ),
                child: _isProcessing
                    ? const Padding(
                        padding: EdgeInsets.all(16),
                        child: CircularProgressIndicator(
                            color: Colors.white, strokeWidth: 2))
                    : const Icon(Icons.camera_alt_rounded,
                        color: Colors.white, size: 30),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
