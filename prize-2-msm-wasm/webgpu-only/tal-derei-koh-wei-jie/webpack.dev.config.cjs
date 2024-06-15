/* eslint-disable @typescript-eslint/no-var-requires */
const path = require("path");
const webpack = require("webpack");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyWebpackPlugin = require('copy-webpack-plugin');
const ForkTsCheckerWebpackPlugin = require("fork-ts-checker-webpack-plugin");
const ESLintPlugin = require("eslint-webpack-plugin");

const config = {
  mode: "development",
  output: {
    publicPath: "/",
  },
  entry: "./src/index.tsx",
  experiments: {
    asyncWebAssembly: true
  },
  module: {
    rules: [
      {
        test: /\.(ts|js)x?$/i,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
          options: {
            presets: [
              "@babel/preset-env",
              "@babel/preset-react",
              "@babel/preset-typescript",
            ],
          },
        },
      },
      {
        test: /\.css$/i,
        exclude: /node_modules/,
        use: ['style-loader', 'css-loader', 'postcss-loader'],
      },
      {
        test: /\.wgsl/i,
        type: 'asset/source',
      },
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
    fallback: {
      "assert": require.resolve("assert"),
      "crypto": require.resolve("crypto-browserify"),
      "stream": require.resolve("stream-browserify"),
      "buffer": require.resolve("buffer/"),
      "path": require.resolve("path-browserify")
    }
  },
  plugins: [
    new webpack.ProvidePlugin({
      Buffer: ['buffer', 'Buffer'],
    }),
    new HtmlWebpackPlugin({
      template: "public/index.html",
    }),
    new webpack.HotModuleReplacementPlugin(),
    new ForkTsCheckerWebpackPlugin({
      async: false,
    }),
    new ESLintPlugin({
      extensions: ["js", "jsx", "ts", "tsx"],
    }),
    new CopyWebpackPlugin({
      patterns: [
        {
          from: path.resolve(__dirname, 'public/test-data'),
          to: path.resolve(__dirname, 'dist/test-data'),
        }
      ]
    }),
    new webpack.DefinePlugin({
      //'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV),
      'process.env.NODE_ENV': JSON.stringify('development'),
      'process.env.NODE_DEBUG': JSON.stringify(process.env.NODE_DEBUG),
      'process.type': JSON.stringify(process.type),
      'process.version': JSON.stringify(process.version),
    })
  ],
  devtool: "inline-source-map",
  devServer: {
    // watchContentBase: true,
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
    static: path.join(__dirname, "build"),
    historyApiFallback: true,
    port: 4040,
    open: false,
    hot: true,
    client: {
      overlay: false
    }
  },
};

const workerConfig = {
  mode: "development",
  devtool: "inline-source-map",
  cache: {
    type: 'filesystem',
    allowCollectingMemory: true
  },
  performance: {
    hints: false
  },
  experiments: {
    asyncWebAssembly: true,
    syncWebAssembly: true,
    topLevelAwait: true
  },
  target: 'web',
  entry: {
    wasmMSM: './src/workers/wasmMSM.ts',
  },
  output: {
    pathinfo: false,
    publicPath: '/',
  },
  resolve: {
    extensions: ['.ts', '.tsx', '.js', '.wasm'],
    alias: {
      shared: path.resolve(__dirname, 'src', 'shared')
    },
    fallback: {
      url: false,
      os: false,
      path: false,
      stream: false,
      crypto: require.resolve("crypto-browserify"),
      http: false,
      https: false,
      buffer: require.resolve('buffer'),
      stream: require.resolve('stream-browserify'),
      assert: require.resolve('assert')
    }
  },
  plugins: [
    new webpack.ProvidePlugin({
      Buffer: ['buffer', 'Buffer']
    }),

    new webpack.ProvidePlugin({
      process: 'process/browser'
    }),
  ],
  module: {
    rules: [
      {
        test: /\.m?js$/i,
        exclude: /node_modules/,
        type: 'javascript/auto'
      },
      {
        test: /\.(ts|js)x?$/i,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
          options: {
            presets: [
              "@babel/preset-env",
              "@babel/preset-typescript",
            ],
          },
        },
      },
    ]
  }
};

module.exports = [config, workerConfig];
