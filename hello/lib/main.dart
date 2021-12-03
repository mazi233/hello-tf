import 'dart:io';

import 'package:flutter/material.dart';
import 'package:grpc/grpc.dart';
import 'package:hello/generated/infer.pbgrpc.dart' as infer;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(title: '图像识别'),
      debugShowCheckedModeBanner: false,
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final ImagePicker _picker = ImagePicker();
  final infer.WebClient _cli = infer.WebClient(ClientChannel(
    '192.168.20.51',
    port: 3001,
    options: const ChannelOptions(credentials: ChannelCredentials.insecure()),
  ));

  var _image;
  var _preds;

  void _pickImage(ImageSource source) async {
    XFile? image =
        await _picker.pickImage(source: source, maxHeight: 800, maxWidth: 800);

    if (image == null) {
      return;
    }

    setState(() {
      _image = File(image.path);
    });

    infer.WebResponse res =
        await _cli.process(infer.WebRequest(image: await image.readAsBytes()));

    setState(() {
      _preds = res.preds;
    });
  }

  @override
  Widget build(BuildContext context) {
    var textStyle = Theme.of(context).textTheme.headline6;
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          // horizontal).
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Container(
              child: _preds != null
                  ? Column(children: [
                      Text(
                        "${_preds[0].name} => ${_preds[0].probability.toStringAsFixed(3)}",
                        style: textStyle,
                      ),
                      Text(
                        "${_preds[1].name} => ${_preds[1].probability.toStringAsFixed(3)}",
                        style: textStyle,
                      ),
                      Text(
                        "${_preds[2].name} => ${_preds[2].probability.toStringAsFixed(3)}",
                        style: textStyle,
                      ),
                    ])
                  : null,
            ),
            Container(
              child: _image != null
                  ? Image.file(
                      _image,
                      width: 400,
                      height: 400,
                    )
                  : const Text('点击下面按钮!'),
            ),
            ButtonBar(
              buttonPadding:
                  const EdgeInsets.symmetric(horizontal: 30, vertical: 10),
              alignment: MainAxisAlignment.center,
              children: [
                TextButton(
                    style: TextButton.styleFrom(
                      padding: const EdgeInsets.all(10.0),
                      primary: Colors.white,
                      backgroundColor: Colors.green,
                      textStyle: const TextStyle(fontSize: 14),
                    ),
                    onPressed: () {
                      _pickImage(ImageSource.camera);
                    },
                    child: const Text('拍照')),
                TextButton(
                    style: TextButton.styleFrom(
                      padding: const EdgeInsets.all(10.0),
                      primary: Colors.white,
                      backgroundColor: Colors.blue,
                      textStyle: const TextStyle(fontSize: 14),
                    ),
                    onPressed: () {
                      _pickImage(ImageSource.gallery);
                    },
                    child: const Text('选择照片')),
              ],
            )
          ],
        ),
      ),
    );
  }
}
