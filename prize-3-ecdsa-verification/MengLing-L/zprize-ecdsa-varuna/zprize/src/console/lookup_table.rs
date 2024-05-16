// Copyright (C) 2024 Mengling LIU, The Hong Kong Polytechnic University
//
// Apache License 2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// MIT License
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the “Software”),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use indexmap::IndexMap;
use snarkvm_console::types::Field;
use snarkvm_console_network::Network;

const DEFAULT_KEY_SIZE: usize = 2;

#[derive(Clone, Debug)]
pub struct LookupTable<N: Network> {
    pub table: IndexMap<[Field<N>; DEFAULT_KEY_SIZE], Field<N>>,
}

impl<N: Network> Default for LookupTable<N> {
    fn default() -> Self {
        Self {
            table: IndexMap::new(),
        }
    }
}

impl<N: Network> LookupTable<N> {
    pub fn fill(&mut self, key: [Field<N>; DEFAULT_KEY_SIZE], val: Field<N>) -> Option<Field<N>> {
        self.table.insert(key, val)
    }

    pub fn lookup(&self, key: &[Field<N>]) -> Option<(usize, &[Field<N>; 2], &Field<N>)> {
        self.table.get_full(key)
    }
}
