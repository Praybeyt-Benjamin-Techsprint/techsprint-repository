import fs from "node:fs/promises";
import path from "node:path";

const repoRoot = path.resolve(".");
const dataDir = path.join(repoRoot, "translation_data");
const outputDir = path.join(repoRoot, "outputs", "sign_translation_tables");
const actionsPath = path.join(repoRoot, "sign_actions.py");

async function loadActions() {
  const source = await fs.readFile(actionsPath, "utf8");
  const match = source.match(/ACTIONS\s*=\s*\[([\s\S]*?)\]/);
  if (!match) {
    throw new Error("Could not find ACTIONS in sign_actions.py");
  }
  return [...match[1].matchAll(/"([^"]+)"/g)].map(([, action], actionId) => ({
    action,
    actionId,
  }));
}

const languages = [
  {
    target_lang: "fil",
    target_locale: "fil-PH",
    language_name: "Filipino (Tagalog)",
    notes: "Use Filipino for broad Philippine national-language output; regional Tagalog wording may vary.",
  },
  {
    target_lang: "ceb",
    target_locale: "ceb-PH",
    language_name: "Cebuano",
    notes: "Validate with speakers from the intended deployment area, e.g. Cebu, Bohol, Mindanao.",
  },
  {
    target_lang: "ilo",
    target_locale: "ilo-PH",
    language_name: "Ilokano",
    notes: "Also written as Ilocano/Iloko in some systems; keep one internal code consistently.",
  },
  {
    target_lang: "pam",
    target_locale: "pam-PH",
    language_name: "Kapampangan",
    notes: "Regional and orthographic variants are common; store alternates as separate variant rows.",
  },
  {
    target_lang: "hil",
    target_locale: "hil-PH",
    language_name: "Hiligaynon",
    notes: "Validate region-specific variants, especially Iloilo versus Negros usage.",
  },
];

const actionMetadata = {
  hello: { output_type: "phrase", canonical_text: "Hello" },
  thank_you: { output_type: "phrase", canonical_text: "Thank you" },
  see_you_later: { output_type: "phrase", canonical_text: "See you later" },
  see: { output_type: "word", canonical_text: "See" },
  you: { output_type: "word", canonical_text: "You" },
  later: { output_type: "word", canonical_text: "Later" },
  yes: { output_type: "word", canonical_text: "Yes" },
  no: { output_type: "word", canonical_text: "No" },
  help: { output_type: "word", canonical_text: "Help" },
  me: { output_type: "word", canonical_text: "Me" },
  father: { output_type: "word", canonical_text: "Father" },
  mother: { output_type: "word", canonical_text: "Mother" },
  abuse: { output_type: "word", canonical_text: "Abuse" },
  please: { output_type: "word", canonical_text: "Please" },
  want: { output_type: "word", canonical_text: "Want" },
  what: { output_type: "word", canonical_text: "What" },
  eat_food: { output_type: "phrase", canonical_text: "Eat food" },
  more: { output_type: "word", canonical_text: "More" },
  go_to: { output_type: "phrase", canonical_text: "Go to" },
  fine: { output_type: "word", canonical_text: "Fine" },
  like: { output_type: "word", canonical_text: "Like" },
  name: { output_type: "word", canonical_text: "Name" },
  meet: { output_type: "word", canonical_text: "Meet" },
  nice: { output_type: "word", canonical_text: "Nice" },
  Sorry: { output_type: "word", canonical_text: "Sorry" },
  where: { output_type: "word", canonical_text: "Where" },
  call: { output_type: "word", canonical_text: "Call" },
};

function titleFromAction(action) {
  return action
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

const actions = (await loadActions()).map(({ action, actionId }) => {
    const metadata = actionMetadata[action] ?? {
      output_type: action.includes("_") ? "phrase" : "word",
      canonical_text: titleFromAction(action),
    };
    return { action, actionId, ...metadata };
  });
const actionSet = new Set(actions.map(({ action }) => action));

const alphabet = "abcdefghijklmnopqrstuvwxyz".split("").map((letter, idx) => ({
  action_id: idx + 27,
  action: letter,
  output_type: "letter",
  canonical_text: letter.toUpperCase(),
  app_output: letter.toUpperCase(),
  notes: "Fingerspelling output; do not translate.",
}));

const draftTranslations = {
  hello: { fil: "Kumusta", ceb: "Kumusta", ilo: "Kumusta", pam: "Kumusta", hil: "Kamusta" },
  thank_you: { fil: "Salamat", ceb: "Salamat", ilo: "Agyamanak", pam: "Salamat", hil: "Salamat" },
  see_you_later: {
    fil: "Kita tayo mamaya",
    ceb: "Magkita ta unya",
    ilo: "Agkita ta inton madamdama",
    pam: "Mikit ta ka pang muli",
    hil: "Magkitaay kita sa ulihi",
  },
  see: { fil: "Tingnan", ceb: "Tan-aw", ilo: "Kitaen", pam: "Lawen", hil: "Tan-aw" },
  you: { fil: "Ikaw", ceb: "Ikaw", ilo: "Sika", pam: "Ika", hil: "Ikaw" },
  later: { fil: "Mamaya", ceb: "Unya", ilo: "Inton madamdama", pam: "Pang muli", hil: "Sa ulihi" },
  yes: { fil: "Oo", ceb: "Oo", ilo: "Wen", pam: "Wa", hil: "Huo" },
  no: { fil: "Hindi", ceb: "Dili", ilo: "Saan", pam: "Ali", hil: "Indi" },
  help: { fil: "Tulong", ceb: "Tabang", ilo: "Tulong", pam: "Saup", hil: "Bulig" },
  me: { fil: "Ako", ceb: "Ako", ilo: "Siak", pam: "Aku", hil: "Ako" },
  father: { fil: "Ama", ceb: "Amahan", ilo: "Ama", pam: "Ibpa", hil: "Amay" },
  mother: { fil: "Ina", ceb: "Inahan", ilo: "Ina", pam: "Ima", hil: "Iloy" },
  abuse: { fil: "Pang-aabuso", ceb: "Pag-abuso", ilo: "Panag-abuso", pam: "Pamanyamantala", hil: "Pag-abuso" },
  please: { fil: "Pakiusap", ceb: "Palihug", ilo: "Pangngaasim", pam: "Pakisabi", hil: "Palihog" },
  want: { fil: "Gusto", ceb: "Gusto", ilo: "Kayat", pam: "Buri", hil: "Gusto" },
  what: { fil: "Ano", ceb: "Unsa", ilo: "Ania", pam: "Nanu", hil: "Ano" },
  eat_food: { fil: "Kumain ng pagkain", ceb: "Kaon og pagkaon", ilo: "Mangan ti taraon", pam: "Mangan pamangan", hil: "Kaon sang pagkaon" },
  more: { fil: "Mas marami", ceb: "Dugang", ilo: "Ad-adu", pam: "Mas dakal pa", hil: "Dugang" },
  go_to: { fil: "Pumunta sa", ceb: "Adto sa", ilo: "Mapan idiay", pam: "Munta king", hil: "Magkadto sa" },
  fine: { fil: "Mabuti", ceb: "Maayo", ilo: "Naimbag", pam: "Mayap", hil: "Maayo" },
  like: { fil: "Gusto", ceb: "Ganahan", ilo: "Kayat", pam: "Buri", hil: "Gusto" },
  name: { fil: "Pangalan", ceb: "Ngalan", ilo: "Nagan", pam: "Lagan", hil: "Ngalan" },
  meet: { fil: "Makilala", ceb: "Makigkita", ilo: "Makipagkita", pam: "Makituki", hil: "Makigkita" },
  nice: { fil: "Maganda", ceb: "Nindot", ilo: "Naimbag", pam: "Mayap", hil: "Nami" },
  Sorry: { fil: "Pasensya", ceb: "Pasayloa ko", ilo: "Pakawanennak", pam: "Patawaran mu ku", hil: "Pasensya" },
  where: { fil: "Saan", ceb: "Asa", ilo: "Sadino", pam: "Nokarin", hil: "Diin" },
  call: { fil: "Tumawag", ceb: "Tawag", ilo: "Umawag", pam: "Tawag", hil: "Tawag" },
};

const phraseRules = [
  ["R001", "help me", "Help me", { fil: "Tulungan mo ako", ceb: "Tabangi ko", ilo: "Tulongannak", pam: "Saupan mu ku", hil: "Buligi ako" }],
  ["R002", "please help me", "Please help me", { fil: "Pakiusap, tulungan mo ako", ceb: "Palihug tabangi ko", ilo: "Pangngaasim, tulongannak", pam: "Pakisabi, saupan mu ku", hil: "Palihog buligi ako" }],
  ["R004", "want eat_food", "I want food", { fil: "Gusto ko ng pagkain", ceb: "Gusto ko og pagkaon", ilo: "Kayatko ti taraon", pam: "Buri ku pamangan", hil: "Gusto ko sang pagkaon" }],
  ["R005", "want more", "I want more", { fil: "Gusto ko pa", ceb: "Gusto pa ko", ilo: "Kayatko pay", pam: "Buri ku pa", hil: "Gusto ko pa" }],
  ["R007", "what name", "What is your name?", { fil: "Ano ang pangalan mo?", ceb: "Unsa imong ngalan?", ilo: "Ania ti naganmo?", pam: "Nanu ing lagyu mu?", hil: "Ano ang ngalan mo?" }],
  ["R008", "what you want", "What do you want?", { fil: "Ano ang gusto mo?", ceb: "Unsa imong gusto?", ilo: "Ania ti kayatmo?", pam: "Nanu ing buri mu?", hil: "Ano ang gusto mo?" }],
  ["R009", "nice meet you", "Nice to meet you", { fil: "Ikinagagalak kitang makilala", ceb: "Nalipay ko nga nakaila nimo", ilo: "Naragsakak nga naam-ammoanka", pam: "Mayap a makilala daka", hil: "Nalipay ako nga nakilala ka" }],
  ["R010", "see you later", "See you later", { fil: "Kita tayo mamaya", ceb: "Magkita ta unya", ilo: "Agkita ta inton madamdama", pam: "Mikit ta ka pang muli", hil: "Magkitaay kita sa ulihi" }],
  ["R011", "see you", "See you", { fil: "Kita tayo", ceb: "Magkita ta", ilo: "Agkita ta", pam: "Mikit ta", hil: "Magkitaay kita" }],
  ["R012", "me fine", "I am fine", { fil: "Mabuti ako", ceb: "Maayo ko", ilo: "Naimbagak", pam: "Mayap ku", hil: "Maayo ako" }],
  ["R013", "father help me", "Father, help me", { fil: "Ama, tulungan mo ako", ceb: "Amahan, tabangi ko", ilo: "Ama, tulongannak", pam: "Ibpa, saupan mu ku", hil: "Amay, buligi ako" }],
  ["R014", "mother help me", "Mother, help me", { fil: "Ina, tulungan mo ako", ceb: "Inahan, tabangi ko", ilo: "Ina, tulongannak", pam: "Ima, saupan mu ku", hil: "Iloy, buligi ako" }],
  ["R015", "abuse help me", "I am being abused. Help me.", { fil: "Inaabuso ako. Tulungan mo ako.", ceb: "Giabuso ko. Tabangi ko.", ilo: "Maab-abusoak. Tulongannak.", pam: "Aabusu ku. Saupan mu ku.", hil: "Ginaabuso ako. Buligi ako." }],
  ["R020", "no abuse me", "Do not abuse me", { fil: "Huwag mo akong abusuhin", ceb: "Ayaw ko abusohi", ilo: "Saan nak nga abusoen", pam: "Ali mu ku abusuan", hil: "Indi mo ako pag-abusuhon" }],
];

const headers = {
  direct: [
    "action_id",
    "action",
    "output_type",
    "canonical_text",
    "source_lang",
    "target_lang",
    "target_locale",
    "region",
    "variant_id",
    "is_primary",
    "translation",
    "register",
    "formality",
    "split",
    "translator_id",
    "reviewer_id",
    "quality_status",
    "notes",
  ],
  phrase: [
    "rule_id",
    "sequence",
    "normalized_sequence",
    "canonical_text",
    "source_lang",
    "target_lang",
    "target_locale",
    "region",
    "variant_id",
    "is_primary",
    "translation",
    "priority",
    "split",
    "translator_id",
    "reviewer_id",
    "quality_status",
    "notes",
  ],
  alphabet: ["action_id", "action", "output_type", "canonical_text", "app_output", "notes"],
  languages: ["target_lang", "target_locale", "language_name", "notes"],
};

function splitForIndex(index) {
  if (index % 10 === 0) return "test";
  if (index % 7 === 0) return "dev";
  return "train";
}

const directRows = actions.flatMap(({ action, actionId, output_type: outputType, canonical_text: canonical }) => {
  return languages.map((lang) => [
    actionId,
    action,
    outputType,
    canonical,
    "en",
    lang.target_lang,
    lang.target_locale,
    "",
    "v1",
    true,
    draftTranslations[action]?.[lang.target_lang] ?? "",
    "neutral",
    "neutral",
    splitForIndex(actionId),
    "AI_DRAFT",
    "",
    "needs_native_review",
    "Draft starter translation; validate with a native speaker before use.",
  ]);
});

const modelPhraseRules = phraseRules.filter(([, sequence]) => {
  return sequence.split(/\s+/).every((action) => actionSet.has(action));
});

const phraseRows = modelPhraseRules.flatMap(([ruleId, sequence, canonical, translations], idx) => {
  return languages.map((lang) => [
    ruleId,
    sequence,
    sequence.trim().toLowerCase().replace(/\s+/g, " "),
    canonical,
    "en",
    lang.target_lang,
    lang.target_locale,
    "",
    "v1",
    true,
    translations[lang.target_lang] ?? "",
    100 - idx,
    splitForIndex(idx + 1),
    "AI_DRAFT",
    "",
    "needs_native_review",
    "Phrase rule should override word-by-word translation when the sequence matches.",
  ]);
});

const languageRows = languages.map((lang) => headers.languages.map((key) => lang[key]));
const alphabetRows = alphabet.map((row) => headers.alphabet.map((key) => row[key]));

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\n\r]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function toCsv(header, rows) {
  return [header, ...rows].map((row) => row.map(csvEscape).join(",")).join("\n") + "\n";
}

function columnLetter(columnNumber) {
  let dividend = columnNumber;
  let columnName = "";
  while (dividend > 0) {
    const modulo = (dividend - 1) % 26;
    columnName = String.fromCharCode(65 + modulo) + columnName;
    dividend = Math.floor((dividend - modulo) / 26);
  }
  return columnName;
}

function writeSheet(workbook, name, header, rows, tableName, widths = {}) {
  const sheet = workbook.worksheets.add(name);
  const matrix = [header, ...rows];
  const endCol = columnLetter(header.length);
  const endRow = matrix.length;
  const range = sheet.getRange(`A1:${endCol}${endRow}`);
  range.values = matrix;
  range.format.font.name = "Aptos";
  range.format.font.size = 10;
  range.format.wrapText = true;
  sheet.getRange(`A1:${endCol}1`).format.fill.color = "#1F4E79";
  sheet.getRange(`A1:${endCol}1`).format.font.color = "#FFFFFF";
  sheet.getRange(`A1:${endCol}1`).format.font.bold = true;
  sheet.getRange(`A1:${endCol}1`).format.rowHeightPx = 30;
  range.format.borders = { preset: "outside", style: "thin", color: "#B7C9D6" };
  range.format.autofitColumns();
  range.format.autofitRows();
  for (const [colNumber, widthPx] of Object.entries(widths)) {
    sheet.getRange(`${columnLetter(Number(colNumber))}:${columnLetter(Number(colNumber))}`).format.columnWidthPx = widthPx;
  }
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  sheet.tables.add(`A1:${endCol}${endRow}`, true, tableName);
  return sheet;
}

async function main() {
  await fs.mkdir(dataDir, { recursive: true });
  await fs.mkdir(outputDir, { recursive: true });

  const translationTables = {
    generated_at: new Date().toISOString(),
    source_actions: path.relative(repoRoot, actionsPath),
    dialects: Object.fromEntries(
      languages.map((lang) => [
        lang.target_lang,
        {
          name: lang.language_name,
          locale: lang.target_locale,
          language: lang.target_locale,
        },
      ]),
    ),
    translations: Object.fromEntries(
      actions.map(({ action }) => [
        action,
        Object.fromEntries(
          languages.map((lang) => [
            lang.target_lang,
            draftTranslations[action]?.[lang.target_lang] ?? "",
          ]),
        ),
      ]),
    ),
  };

  await fs.writeFile(path.join(dataDir, "direct_translations.csv"), toCsv(headers.direct, directRows), "utf8");
  await fs.writeFile(path.join(dataDir, "phrase_rules.csv"), toCsv(headers.phrase, phraseRows), "utf8");
  await fs.writeFile(path.join(dataDir, "alphabet_outputs.csv"), toCsv(headers.alphabet, alphabetRows), "utf8");
  await fs.writeFile(path.join(dataDir, "languages.csv"), toCsv(headers.languages, languageRows), "utf8");
  await fs.writeFile(
    path.join(repoRoot, "translation_tables.json"),
    JSON.stringify(translationTables, null, 2) + "\n",
    "utf8",
  );
  await fs.writeFile(
    path.join(outputDir, "translation_tables.json"),
    JSON.stringify(translationTables, null, 2) + "\n",
    "utf8",
  );

  let artifactTool = null;
  try {
    artifactTool = await import("@oai/artifact-tool");
  } catch (error) {
    await fs.writeFile(
      path.join(outputDir, "build_summary.json"),
      JSON.stringify(
        {
          translationTablePath: path.join(repoRoot, "translation_tables.json"),
          sourceActions: path.relative(repoRoot, actionsPath),
          modelLabels: actions.length,
          csvDirectory: dataDir,
          directRows: directRows.length,
          phraseRows: phraseRows.length,
          filteredPhraseRules: phraseRules.length - modelPhraseRules.length,
          alphabetRows: alphabetRows.length,
          languageRows: languageRows.length,
          xlsxSkipped: true,
          xlsxSkippedReason: error.message,
        },
        null,
        2,
      ),
      "utf8",
    );
    return;
  }

  const { SpreadsheetFile, Workbook } = artifactTool;

  const workbook = Workbook.create();

  const readme = workbook.worksheets.add("README");
  readme.showGridLines = false;
  readme.getRange("A1:B1").merge();
  readme.getRange("A1").values = [["Sign Language Translation Dataset Starter"]];
  readme.getRange("A1").format.font.bold = true;
  readme.getRange("A1").format.font.size = 18;
  readme.getRange("A1").format.fill.color = "#1F4E79";
  readme.getRange("A1").format.font.color = "#FFFFFF";
  readme.getRange("A3:B10").values = [
    ["Purpose", "Starter tables for mapping recognized sign actions to Philippine language outputs."],
    ["Workflow", "Use DirectTranslations for single model actions. Use PhraseRules when multiple recognized actions form a natural phrase."],
    ["Review status", "All starter translations are marked needs_native_review. Replace AI_DRAFT with translator IDs after human validation."],
    ["Letters", "Alphabet signs are in AlphabetOutputs and should usually be displayed directly, not translated."],
    ["Recommended app flow", "recognized actions -> phrase-rule lookup -> direct translation fallback -> display selected target language"],
    ["Exports", "translation_tables.json plus CSV files in translation_data/."],
    ["Caution", "Do not deploy emergency or abuse-related outputs until reviewed by native speakers and accessibility stakeholders."],
    ["Date created", "2026-06-26"],
  ];
  readme.getRange("A3:A10").format.font.bold = true;
  readme.getRange("A3:B10").format.wrapText = true;
  readme.getRange("A3:B10").format.borders = { preset: "outside", style: "thin", color: "#B7C9D6" };
  readme.getRange("A:A").format.columnWidthPx = 150;
  readme.getRange("B:B").format.columnWidthPx = 720;
  readme.getRange("A3:B10").format.autofitRows();

  writeSheet(workbook, "Languages", headers.languages, languageRows, "LanguagesTable", { 1: 90, 2: 110, 3: 190, 4: 620 });
  writeSheet(workbook, "DirectTranslations", headers.direct, directRows, "DirectTranslationsTable", {
    2: 125,
    4: 175,
    11: 230,
    17: 150,
    18: 340,
  });
  writeSheet(workbook, "PhraseRules", headers.phrase, phraseRows, "PhraseRulesTable", {
    2: 155,
    4: 230,
    11: 260,
    17: 360,
  });
  writeSheet(workbook, "AlphabetOutputs", headers.alphabet, alphabetRows, "AlphabetOutputsTable", {
    4: 140,
    6: 300,
  });

  const checks = [];
  checks.push(await workbook.inspect({ kind: "table", sheetId: "DirectTranslations", range: "A1:R8", include: "values", maxChars: 4000 }));
  checks.push(await workbook.inspect({ kind: "table", sheetId: "PhraseRules", range: "A1:Q8", include: "values", maxChars: 4000 }));
  checks.push(await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "formula error scan" }));

  for (const sheetName of ["README", "Languages", "DirectTranslations", "PhraseRules", "AlphabetOutputs"]) {
    const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
    const previewBytes = new Uint8Array(await preview.arrayBuffer());
    await fs.writeFile(path.join(outputDir, `${sheetName}.png`), previewBytes);
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  const workbookPath = path.join(outputDir, "sign_language_translation_tables.xlsx");
  await output.save(workbookPath);

  await fs.writeFile(
    path.join(outputDir, "build_summary.json"),
    JSON.stringify(
      {
        workbookPath,
        translationTablePath: path.join(repoRoot, "translation_tables.json"),
        sourceActions: path.relative(repoRoot, actionsPath),
        modelLabels: actions.length,
        csvDirectory: dataDir,
        directRows: directRows.length,
        phraseRows: phraseRows.length,
        filteredPhraseRules: phraseRules.length - modelPhraseRules.length,
        alphabetRows: alphabetRows.length,
        languageRows: languageRows.length,
        checks: checks.map((check) => check.ndjson).join("\n"),
      },
      null,
      2,
    ),
    "utf8",
  );
}

await main();
